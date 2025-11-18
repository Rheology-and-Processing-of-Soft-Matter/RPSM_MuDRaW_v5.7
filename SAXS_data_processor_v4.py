import sys
import os
import re
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import P2468OAS_v3_3 as hop
from scipy.special import legendre
import json

# --- Radial normalization helpers ---
from typing import Tuple, Optional

def _looks_like_header(line: str) -> bool:
    """Return True if the first line contains non-numeric tokens (e.g., 'ang,f1,...')."""
    stripped = (line or "").strip()
    if not stripped:
        return True
    tokens = stripped.replace(",", " ").split()
    if not tokens:
        return True
    try:
        float(tokens[0])
        return False
    except ValueError:
        return True

def _scan_header_info(path: str, max_lines: int = 20) -> tuple[str, int]:
    """
    Inspect the file and return (header_line, skiprows) where skiprows counts any
    blank/comment/header rows preceding numeric data.
    """
    header_line = ""
    skiprows = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                raw = line.rstrip("\r\n")
                stripped = raw.strip()
                if not stripped:
                    skiprows += 1
                    continue
                if stripped.startswith("#"):
                    header_line = stripped
                    skiprows += 1
                    continue
                if _looks_like_header(stripped):
                    header_line = stripped
                    skiprows += 1
                    continue
                # first non-empty, non-header, non-comment line reached
                break 
    except Exception:
        header_line = ""
        skiprows = 0
    return header_line, skiprows

def _load_dat_matrix(path: str) -> np.ndarray:
    """Robustly load a .dat file that may or may not have a header row."""
    _, skiprows = _scan_header_info(path)
    try:
        return np.loadtxt(path, skiprows=skiprows)
    except ValueError:
        return np.loadtxt(path, delimiter=",", skiprows=skiprows)

def _derive_radial_headers(header_line: str, column_count: int) -> list[str]:
    """Return a robust set of column labels for the radial matrix."""
    cleaned = header_line.split("#", 1)[-1].strip() if header_line else ""
    tokens = [tok.strip() for tok in cleaned.replace(",", " ").split() if tok.strip()]
    headers: list[str] = []
    if tokens:
        headers = tokens[:column_count]
    if len(headers) < column_count:
        headers.extend([f"frame_{i+1}" for i in range(len(headers), column_count)])
    if headers:
        headers[0] = "q"
    else:
        headers = ["q"] + [f"frame_{i+1}" for i in range(column_count - 1)]
    return headers

def _build_header_line(first_label: str, data_cols: int, prefix: str) -> str:
    labels = [first_label] + [f"{prefix}{i+1}" for i in range(data_cols)]
    return " ".join(labels)

def _env_disables_plots() -> bool:
    val = os.environ.get("MUDRAW_SAXS_NO_PLOTS", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}

def _plots_enabled(requested: bool) -> bool:
    return bool(requested) and not _env_disables_plots()

def _write_matrix_with_header(path: str, data: np.ndarray, header_line: Optional[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        if header_line:
            fh.write(header_line.strip() + "\n")
        for row in data:
            fh.write("\t".join(f"{val:.10e}" for val in row) + "\n")

def _average_columns_if_needed(
    matrix: np.ndarray,
    first_label: str,
    header_prefix: str,
    group_size: int = 4,
    trigger_total_cols: int = 117,
    write_back_path: Optional[str] = None,
) -> tuple[np.ndarray, Optional[str], bool]:
    """Average every `group_size` columns (excluding the first) if total columns match trigger."""
    total_cols = matrix.shape[1]
    if total_cols != trigger_total_cols:
        return matrix, None, False
    data_block = matrix[:, 1:]
    if data_block.shape[1] % group_size != 0:
        return matrix, None, False
    n_groups = data_block.shape[1] // group_size
    averaged = data_block.reshape(matrix.shape[0], n_groups, group_size).mean(axis=2)
    combined = np.hstack([matrix[:, [0]], averaged])
    header_line = _build_header_line(first_label, n_groups, header_prefix)
    if write_back_path:
        try:
            _write_matrix_with_header(write_back_path, combined, header_line)
            print(f"[SAXS] Averaged {data_block.shape[1]} columns into {n_groups} for {write_back_path}")
        except Exception as exc:
            print(f"[SAXS] Warning: failed to rewrite averaged file {write_back_path}: {exc}")
    return combined, header_line, True

def _export_radial_csvs(radial_path: str, sample_tag: str, results_path: str) -> Tuple[Optional[str], Optional[str], list[str], Optional[str]]:
    """
    Read the radial integration matrix and emit:
      1) A matrix CSV (all frames preserved)
      2) A flattened CSV (Intensity, Frames, q)
    Returns (matrix_csv, flat_csv, headers, error_message)
    """
    header_line, skiprows = _scan_header_info(radial_path)
    try:
        data = np.loadtxt(radial_path, dtype=float, comments="#", skiprows=skiprows)
    except Exception:
        try:
            data = np.loadtxt(radial_path, dtype=float, comments="#", delimiter=",", skiprows=skiprows)
        except Exception as exc:
            return None, None, [], f"Failed to parse radial data: {exc}"

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        return None, None, [], "Radial file appears to be empty."

    data, averaged_header, averaged = _average_columns_if_needed(
        data, first_label="q", header_prefix="frame_", write_back_path=radial_path
    )
    if averaged and averaged_header:
        header_line = averaged_header

    column_count = data.shape[1]
    headers = _derive_radial_headers(header_line, column_count)

    matrix_df = pd.DataFrame(data, columns=headers)
    matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")
    if "q" in matrix_df.columns:
        matrix_df = matrix_df.dropna(subset=["q"])
    matrix_df = matrix_df.reset_index(drop=True)
    matrix_csv = os.path.join(results_path, f"SAXS_radial_matrix_{sample_tag}.csv")
    matrix_df.to_csv(matrix_csv, index=False)

    flat_csv: Optional[str] = None
    if column_count >= 2:
        numeric_data = matrix_df.to_numpy()
        q_vals = numeric_data[:, 0]
        frame_block = numeric_data[:, 1:]
        n_q = q_vals.shape[0]
        n_frames = frame_block.shape[1] if frame_block.ndim > 1 else 1
        if n_frames == 1 and frame_block.ndim == 1:
            frame_block = frame_block.reshape(-1, 1)
        intensities = frame_block.reshape(-1, order="F")
        q_repeat = np.tile(q_vals, n_frames)
        frame_ids = np.repeat(np.arange(1, n_frames + 1), n_q)
        flat_df = pd.DataFrame(
            {
                "Q": q_repeat,
                "Frame_rad": frame_ids,
                "I_rad": intensities,
            }
        )
        flat_csv = os.path.join(results_path, f"SAXS_radial_flat_{sample_tag}.csv")
        flat_df.to_csv(flat_csv, index=False)

    return matrix_csv, flat_csv, headers, None

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    """Resolve the *reference* folder robustly.

    Walks up the path to find known modality markers ("PLI", "PI", "SAXS", "Rheology").
    The *reference* folder is the parent directory of the modality folder.
    Fallback: two levels up from the provided path (legacy behavior).
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    abspath = os.path.abspath(path)
    parts = _split_parts(abspath)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            # Reconstruct with a leading slash to preserve absolute path
            reference = os.path.join(os.sep, *parts[:i])
            return reference

    # Fallback: two levels up from the provided absolute path
    return os.path.dirname(os.path.dirname(abspath))


def _extract_scan_range(name: str):
    for pat in (r"scan_(\d+)-(\d+)", r"(\d{3,})-(\d{3,})"):
        m = re.search(pat, name)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                pass
    return (None, None)


def _find_radial_for_azimuthal(azi_path: str) -> str | None:
    p = Path(azi_path)
    folder = p.parent
    if not folder.exists():
        return None
    lo, hi = _extract_scan_range(p.name)
    cands = sorted(q for q in folder.glob("rad_saxs*") if q.is_file())
    if not cands:
        cands = sorted(q for q in folder.glob("*rad*.*") if q.is_file())
    if not cands:
        return None
    if lo is not None and hi is not None:
        scored = []
        for q in cands:
            qlo, qhi = _extract_scan_range(q.name)
            score = int(qlo == lo) + int(qhi == hi)
            scored.append((score, q))
        scored.sort(key=lambda t: (-t[0], t[1].name))
        return str(scored[0][1].resolve())
    return str(cands[0].resolve())


def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    processed_root = os.path.abspath(processed_root)
    os.makedirs(processed_root, exist_ok=True)
    return processed_root

from common import update_json_file

try:
    import plotly.express as px
except ImportError:
    print("Error: The 'plotly' module is not installed. Please install it by running 'pip install plotly'.")
    sys.exit(1)


def SAXS_data_processor(path_name, sample_name, smoo_, sigma_zero, lower_limit, upper_limit, plots=True, method="fitting", mirror=False):
    print(f"Processing SAXS data with parameters: {path_name}, {sample_name}, {smoo_}, {sigma_zero}, {lower_limit}, {upper_limit}, method={method}, mirror={mirror}")

    # Convert limits to degrees
    limit_deg = [lower_limit, upper_limit]
    
    # Define the results path
    processed_folder = get_unified_processed_folder(path_name)
    results_path = processed_folder
    print(f"Unified _Processed folder for SAXS outputs: {processed_folder}")
    
    # Get the list of folders in the root directory
    folder_list = [
        f for f in os.listdir(path_name)
        if os.path.isdir(os.path.join(path_name, f))
        and not f.startswith('.')
        and not f.startswith('_Processed')
        and not f.startswith('_output')
    ]
    paths = [os.path.join(path_name, folder) for folder in folder_list]

    if not folder_list:
        print("No subfolders found. Searching for .dat files directly in the provided path.")
        paths = [path_name]

    print(f"Found folders: {folder_list}")
    plot_enabled = _plots_enabled(plots)
    
    for path in paths:
        file_names = [f for f in os.listdir(path) if f.endswith(".dat")]
        print(f"Processing folder: {path}, found files: {file_names}")
        
        for file_name in file_names:
            load_location = os.path.join(path, file_name)
            sample_name2 = f"{sample_name}_{file_name.split('.')[0]}"
            print(f"Loading file: {load_location}")
            radial_file = _find_radial_for_azimuthal(load_location)
            radial_copy_path = None
            radial_csv_abs = None
            radial_csv_rel = None
            radial_matrix_csv_abs = None
            radial_matrix_csv_rel = None
            radial_flat_csv_abs = None
            radial_flat_csv_rel = None
            radial_headers: list[str] = []
            if radial_file:
                print(f"Found radial integration file: {radial_file}")
            else:
                print("No radial integration file located for this azimuthal dataset.")
            try:
                file_raw = _load_dat_matrix(load_location)
            except Exception as e:
                print(f"Failed to load file: {load_location}. Error: {e}")
                continue
            file_raw, _, averaged = _average_columns_if_needed(
                file_raw,
                first_label="ang",
                header_prefix="f",
                write_back_path=load_location,
            )
            if averaged:
                print(f"[SAXS] Azimuthal data averaged to {file_raw.shape[1]} columns for {load_location}")
            
            n_file, m_file = file_raw.shape
            print(f"File shape: {n_file} x {m_file}")
            
            n_frames = m_file - 1
            angles = file_raw[:, 0]
            file_part = file_raw[:, 1:]
            flattened_raw = file_part.flatten(order='F')
            frames = np.repeat(np.arange(n_frames), n_file).reshape(-1, 1)
            angle_flat = np.tile(angles, n_frames).reshape(-1, 1)
            flat_data = np.hstack((flattened_raw.reshape(-1, 1), frames, angle_flat))

            
            
            # Process the data
            limits = np.round(np.array(limit_deg) * n_file / 360)
            print(f"Processing data with limits: {limits}")
            try:
                P2468OAS = hop.P2468OAS_v3_3(
                    method,
                    mirror,
                    angles,
                    file_part,
                    limits,
                    smoo_,
                    sigma_zero,
                    plot=plot_enabled,
                    window_title=f"SAXS — {sample_name2}" if plot_enabled else None,
                )
            except Exception as e:
                continue
            
            m_out, n_out = P2468OAS.shape
            print(f"Processed data shape: {m_out} x {n_out}")
            
            # Save the processed data
            saxs_files = []
            if radial_file:
                try:
                    suffix = Path(radial_file).suffix or ".dat"
                    radial_copy_path = os.path.join(results_path, f"SAXS_radial_{sample_name2}{suffix}")
                    shutil.copy2(radial_file, radial_copy_path)
                    print(f"Copied radial integration to: {radial_copy_path}")
                except Exception as e:
                    print(f"Failed to copy radial integration file: {e}")
                    radial_copy_path = None

                if radial_copy_path:
                    try:
                        matrix_csv, flat_csv, headers, err = _export_radial_csvs(
                            radial_copy_path,
                            sample_name2,
                            results_path,
                        )
                        if err:
                            print(f"Radial export warning: {err}")
                        if matrix_csv and os.path.exists(matrix_csv):
                            radial_matrix_csv_abs = matrix_csv
                            radial_matrix_csv_rel = os.path.relpath(matrix_csv, processed_folder)
                            radial_csv_abs = matrix_csv  # legacy alias for UI
                            radial_csv_rel = radial_matrix_csv_rel
                            print(f"Wrote radial matrix CSV: {matrix_csv}")
                        if flat_csv and os.path.exists(flat_csv):
                            radial_flat_csv_abs = flat_csv
                            radial_flat_csv_rel = os.path.relpath(flat_csv, processed_folder)
                            print(f"Wrote radial flat CSV: {flat_csv}")
                        if headers:
                            radial_headers = headers
                    except Exception as e:
                        print(f"Failed to normalize radial file to CSV: {e}")

            export_path = os.path.join(results_path, f"SAXS_1_{sample_name2}_export.csv")
            column_names = ["P2", "P4", "P6", "OAS"]
            pd.DataFrame(P2468OAS, columns=column_names).to_csv(export_path, index=False, header=True)
            saxs_files.append(export_path)
            print(f"Saved processed data to: {export_path}")

            # Save the raw flattened data
            raw_flat_export_path = os.path.join(results_path, f"SAXS_2_{sample_name2}_raw_flat.csv")
            column_names = ["Intensity", "Frames", "Angles"]
            pd.DataFrame(flat_data, columns=column_names).to_csv(raw_flat_export_path, index=False, header=True)
            saxs_files.append(raw_flat_export_path)
            print(f"Saved raw flattened data to: {raw_flat_export_path}")

            # Include radial exports alongside azimuthal outputs for downstream tools
            if radial_matrix_csv_abs and os.path.exists(radial_matrix_csv_abs):
                saxs_files.append(radial_matrix_csv_abs)
            if radial_flat_csv_abs and os.path.exists(radial_flat_csv_abs):
                saxs_files.append(radial_flat_csv_abs)

            print(f"SAXS results saved to unified folder: {results_path}")

            # Write a per-file unified JSON (unique name per input file)
            json_path = os.path.join(processed_folder, f"_output_SAXS_{sample_name2}.json")
            # Derive robust relative suffixes with os.path.relpath
            rel_suffixes = [os.path.relpath(p, processed_folder) for p in saxs_files]
            azimuthal_file = load_location
            azimuthal_file_rel = (
                os.path.relpath(azimuthal_file, processed_folder)
                if os.path.exists(azimuthal_file) else None
            )
            radial_file_rel = (
                os.path.relpath(radial_file, processed_folder)
                if radial_file and os.path.exists(radial_file) else None
            )
            radial_copy_rel = (
                os.path.relpath(radial_copy_path, processed_folder)
                if radial_copy_path and os.path.exists(radial_copy_path) else None
            )
            radial_matrix_csv_rel = (
                os.path.relpath(radial_matrix_csv_abs, processed_folder)
                if radial_matrix_csv_abs and os.path.exists(radial_matrix_csv_abs) else None
            )
            radial_flat_csv_rel = (
                os.path.relpath(radial_flat_csv_abs, processed_folder)
                if radial_flat_csv_abs and os.path.exists(radial_flat_csv_abs) else None
            )

            payload = {
                "sample_name": sample_name2,
                "csv_outputs": saxs_files,
                "csv_outputs_rel": rel_suffixes,
                "azimuthal_file": azimuthal_file,
                "azimuthal_file_rel": azimuthal_file_rel,
                "radial_file": radial_file,
                "radial_file_rel": radial_file_rel,
                # deterministic copy inside _Processed/SAXS
                "radial_copy": radial_copy_path,
                "radial_copy_rel": radial_copy_rel,
                # normalized CSV exports (matrix + flattened)
                "radial_csv": radial_csv_abs,  # legacy name pointing to matrix CSV
                "radial_csv_rel": radial_csv_rel,
                "radial_matrix_csv": radial_matrix_csv_abs,
                "radial_matrix_csv_rel": radial_matrix_csv_rel,
                "radial_flat_csv": radial_flat_csv_abs,
                "radial_flat_csv_rel": radial_flat_csv_rel,
                "radial_headers": radial_headers,
            }
            try:
                with open(json_path, "w") as jf:
                    json.dump(payload, jf, indent=2)
                print(f"Wrote unified JSON: {json_path}")
            except Exception as e:
                print(f"Failed to write JSON {json_path}: {e}")

def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Process SAXS azimuthal/radial datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path_name", help="Folder containing the sample subfolders or .dat files.")
    parser.add_argument("sample_name", help="Sample prefix for outputs.")
    parser.add_argument("smoo_", type=float, help="Smoothing value.")
    parser.add_argument("sigma_zero", type=float, help="Sigma zero.")
    parser.add_argument("lower_limit", type=float, help="Lower angular limit (deg).")
    parser.add_argument("upper_limit", type=float, help="Upper angular limit (deg).")
    parser.add_argument("--fast", action="store_true", help="Skip plotting windows.")
    parser.add_argument("--no-plots", action="store_true", dest="no_plots", help="Alias for --fast.")
    parser.add_argument(
        "--method",
        choices=["fitting", "direct"],
        default="fitting",
        help="HoP computation method.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror azimuthal data prior to processing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    plots = not (args.fast or args.no_plots)
    SAXS_data_processor(
        args.path_name,
        args.sample_name,
        args.smoo_,
        args.sigma_zero,
        args.lower_limit,
        args.upper_limit,
        plots=plots,
        method=args.method,
        mirror=args.mirror,
    )
