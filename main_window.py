# --- Rheology CSV launcher ---
def process_rheology_file(csv_path, mode="triggered", steady_sec=10.0):
    try:
        script_path = os.path.join(base_path, "Read_viscosity_v3.py")  # swap to v4 when ready
        if not os.path.isfile(script_path):
            messagebox.showerror("Rheology", f"Viscosity reader not found:\n{script_path}")
            return
        args = [sys.executable, script_path, csv_path,
                "--mode", mode,
                "--steady-sec", str(steady_sec)]
        print("[Rheology] Launching viscosity reader:", args)
        subprocess.Popen(args, cwd=os.path.dirname(csv_path))
    except Exception as e:
        messagebox.showerror("Rheology", f"Failed to launch reader:\n{e}")

# --- Rheology reader dialog (mode / steady-sec / threshold) ---

def _rheo_session_path(reference_folder: str) -> str:
    try:
        sp = os.path.join(reference_folder, "_Processed", "_rheo_reader_session.json")
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        return sp
    except Exception:
        return os.path.join(reference_folder, "_rheo_reader_session.json")


def _rheo_load_session(reference_folder: str) -> dict:
    sp = _rheo_session_path(reference_folder)
    try:
        if os.path.isfile(sp):
            with open(sp, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _rheo_save_session(reference_folder: str, data: dict):
    sp = _rheo_session_path(reference_folder)
    try:
        with open(sp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        print("[Rheology] session saved →", sp)
    except Exception as e:
        print("[Rheology] session save failed:", e)


def open_rheology_reader_dialog(parent, csv_path):
    """Small options dialog before launching the viscosity reader."""
    # Per-dataset persistence: use the current reference folder if available
    reference_folder = getattr(parent, "current_folder", os.path.dirname(os.path.dirname(csv_path)))
    sess = _rheo_load_session(reference_folder)

    dlg = tk.Toplevel(parent)
    dlg.title("Read viscosity — options")
    try:
        x, y = parent.get_next_window_position(360, 150)
        dlg.geometry(f"360x150+{x}+{y}")
        parent.register_window_close(dlg, (x, y, 360, 150))
    except Exception:
        dlg.geometry("360x150")

    frame = ttk.Frame(dlg)
    frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    # Mode
    ttk.Label(frame, text="Mode:").grid(row=0, column=0, sticky="w")
    mode_var = tk.StringVar(value=sess.get("mode", "triggered"))
    modes = ttk.Frame(frame); modes.grid(row=0, column=1, columnspan=2, sticky="w")
    ttk.Radiobutton(modes, text="Triggered",     value="triggered",    variable=mode_var).pack(side=tk.LEFT)
    ttk.Radiobutton(modes, text="Non-triggered", value="nontriggered", variable=mode_var).pack(side=tk.LEFT, padx=(8,0))
    ttk.Radiobutton(modes, text="Other",         value="other",        variable=mode_var).pack(side=tk.LEFT, padx=(8,0))

    # Steady seconds (will be hidden for Triggered mode)
    ttk.Label(frame, text="Steady window (s):").grid(row=1, column=0, sticky="w", pady=(8,0))
    steady_var = tk.DoubleVar(value=float(sess.get("steady_sec", 10.0)))
    steady_entry = ttk.Entry(frame, textvariable=steady_var, width=8)
    steady_entry.grid(row=1, column=1, sticky="w", pady=(8,0))




    # Buttons
    btns = ttk.Frame(dlg); btns.pack(fill=tk.X, padx=12, pady=10)

    def _run():
        data = {
            "mode": mode_var.get(),
            "steady_sec": float(steady_var.get()),
        }
        _rheo_save_session(reference_folder, data)
        try:
            process_rheology_file(
                csv_path,
                mode=mode_var.get(),
                steady_sec=steady_var.get(),
            )
        except Exception as e:
            messagebox.showerror("Rheology", f"Failed to launch reader:\n{e}")
        dlg.destroy()

    ttk.Button(btns, text="Run", command=_run).pack(side=tk.RIGHT)
    ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0,8))
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import sys
import subprocess
import tkinter.simpledialog
import re
import importlib
import importlib.util as _importlib_util
from PLI_data_processor import open_maltese_cross_analyzer

# --- Path hygiene (ensure absolute macOS paths) ---

def _abs_path(p: str) -> str:
    """Expanduser + abspath and enforce a leading slash on macOS."""
    if not isinstance(p, str):
        return p
    p2 = os.path.abspath(os.path.expanduser(p))
    if p2 and not p2.startswith(os.sep):
        p2 = os.sep + p2
    return p2

def _clean_user_path(p: str) -> str:
    """If a repo prefix was concatenated before '/Users/...', trim to the real user path.
    Keeps leading slash; otherwise returns the original.
    """
    if not isinstance(p, str):
        return p
    p2 = _abs_path(p)
    needle = "/Users/"
    j = p2.find(needle)
    if j <= 0:
        return p2
    k = p2.find(needle, j + 1)
    if k != -1:
        fixed = p2[k:]
        if not fixed.startswith(os.sep):
            fixed = os.sep + fixed
        print(f"[MAIN] sanitize: trimmed misjoined path\n  in : {p2}\n  out: {fixed}")
        return fixed
    return p2

def _natural_key(s: str):
    """Natural sort key: 'img2' < 'img10'"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# --- Helper to extract folder name from a path ---
def _folder_name(p: str) -> str:
    """Return only the last folder name of a path, robust to trailing slashes."""
    if not isinstance(p, str):
        return p
    p2 = _abs_path(p).rstrip(os.sep)
    return os.path.basename(p2)

# Import PLI helper functions
from _Helpers.PLI_helper import (
    extract_space_time_pli,
    load_intervals_pli,
    rescale_space_time_pli,
    list_raw_st_images,
    list_stitched_outputs,
    process_scaled_outputs_pli,
    list_unscaled_temp_stitched_outputs,
)
# Removed import of extract_space_time_interactive

# Add the Helpers directory to the Python path

from common import confirm_clear_all, update_actions_list, add_action, load_renamed_samples, load_last_folder_internal, save_last_folder, rename_sample, set_global_log_widget
from _Helpers.SAXS_helper import get_last_used_saxs_parameters, engage_saxs
sys.path.append(os.path.join(os.path.dirname(__file__), '_Helpers'))
from Rheology_helper import process_all_rheology_files
from write_template_window import WriteTemplateWindow  # Ensure this import is present
from _Helpers.PI_helper import process_sample_PI


def _load_batch_window_class():
    """Try normal import; if that fails, load BatchProcessingWindow.py by path next to main_window.py."""
    last_error = None
    try:
        from BatchProcessingWindow import BatchProcessingWindow  # type: ignore
        return BatchProcessingWindow, None
    except Exception as exc:
        last_error = exc
    base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base, "BatchProcessingWindow.py")
    if not os.path.isfile(candidate):
        return None, last_error
    spec = _importlib_util.spec_from_file_location("BatchProcessingWindow", candidate)
    if not spec or not spec.loader:
        return None, last_error
    mod = _importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return getattr(mod, "BatchProcessingWindow", None), None
    except Exception as exc:
        return None, exc


 # Load configuration safely
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
try:
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    messagebox.showwarning("Configuration missing", f"Could not find config.json at {config_path}. Using defaults.")
    config = {}
except json.JSONDecodeError as e:
    messagebox.showwarning("Configuration error", f"Error reading config.json:\n{e}")
    config = {}

base_path = os.path.dirname(os.path.abspath(__file__))

# === Unified _Processed folder handling (reference-folder scoped) ===
def get_processed_root(reference_folder: str) -> str:
    """
    Return the unified _Processed folder that lives directly under the selected reference folder.
    """
    reference_folder = os.path.abspath(os.path.expanduser(reference_folder))
    return os.path.join(reference_folder, "_Processed")

def ensure_processed_root(reference_folder: str) -> str:
    """
    Ensure the unified _Processed folder exists under the selected reference folder.
    """
    processed = get_processed_root(reference_folder)
    os.makedirs(processed, exist_ok=True)
    return processed

def find_deepest_subfolder(root_dir: str) -> str:
    """
    Find the deepest subfolder under root_dir. If none exists, return root_dir.
    Depth is measured by path length; ties fall back to the first encountered.
    """
    deepest_path = root_dir
    deepest_len = len(root_dir.split(os.sep))
    for current_root, dirs, _files in os.walk(root_dir):
        for d in dirs:
            cand = os.path.join(current_root, d)
            depth = len(cand.split(os.sep))
            if depth > deepest_len:
                deepest_len = depth
                deepest_path = cand
    return deepest_path

def index_output_jsons(processed_root: str) -> dict:
    """
    Scan the unified _Processed folder and return a mapping of modality -> list of output json files.
    Accept both 'output_*' and legacy '_output_*' file names.
    """
    outputs = {"PI": [], "PLI": [], "SAXS": [], "Rheology": [], "Other": []}
    if not os.path.isdir(processed_root):
        return outputs
    for fname in os.listdir(processed_root):
        if not fname.lower().endswith(".json"):
            continue
        if not (fname.startswith("output_") or fname.startswith("_output_")):
            continue
        fpath = os.path.join(processed_root, fname)
        low = fname.lower()
        if "pli" in low:
            outputs["PLI"].append(fpath)
        elif "saxs" in low:
            outputs["SAXS"].append(fpath)
        elif "pi" in low:
            outputs["PI"].append(fpath)
        elif ("rheo" in low) or ("rheology" in low) or ("viscosity" in low):
            outputs["Rheology"].append(fpath)
        else:
            outputs["Other"].append(fpath)
    for k in outputs:
        outputs[k].sort()
    return outputs

standard_subfolders = {
    'PI': 'PI',
    'SAXS': 'SAXS',
    'Rheology': 'Rheology',
    'PLI': 'PLI'
}

last_folder_file = "last_folder.txt"
performed_actions_file = "performed_actions.json"
stored_geometry_centering = {}
stored_triggers = {}
performed_actions = {}

home_dir = os.path.expanduser("~")
last_selections_file = os.path.join(home_dir, "last_selections.json")
last_preset_values_file = os.path.join(home_dir, "last_preset_values.json")

class SAXSParameterDialog(tk.simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Smoothing factor (smoo_)").grid(row=0)
        tk.Label(master, text="Sigma zero").grid(row=1)
        tk.Label(master, text="Lower limit (degrees)").grid(row=2)
        tk.Label(master, text="Upper limit (degrees)").grid(row=3)

        self.e1 = tk.Entry(master)
        self.e2 = tk.Entry(master)
        self.e3 = tk.Entry(master)
        self.e4 = tk.Entry(master)

        self.e1.insert(0, "0.04")
        self.e2.insert(0, "0.5")
        self.e3.insert(0, "1")
        self.e4.insert(0, "180")
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.e4.grid(row=3, column=1)

        return self.e1

    def apply(self):
        self.result = {
            "smoo_": float(self.e1.get()),
            "sigma_zero": float(self.e2.get()),
            "lower_limit": float(self.e3.get()),
            "upper_limit": float(self.e4.get())
        }

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MuDRaW — Material Processing Suite")
        self.geometry("650x500")
        self.minsize(600, 480)
        self.open_windows = []  # Stores rectangles (x, y, w, h) to prevent overlap
        self.current_folder = None
        self.processed_root = None
        self.available_outputs = {"PI": [], "PLI": [], "SAXS": [], "Rheology": [], "Other": []}

        self.main_frame = ttk.Frame(self, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mode_specs = [
            ("SAXS", "SAXS", self.open_saxs_folder),
            ("Rheo", "Rheology", self.open_rheology_folder),
            ("PLI", "PLI", self.open_pli_folder),
            ("PI", "PI", self.open_pi_folder),
            ("M/TC DMA", None, None),
            ("Tribo", None, None),
        ]
        self.mode_buttons = {}

        self.last_window_position = (None, None)

        self._build_header()
        self._build_mode_buttons()
        self._build_utility_buttons()

        self.log_text = tk.Text(self.content_frame, height=8, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        self.log_text.insert(tk.END, "Welcome to MuDRaW.\n")
        self.log_text.config(state=tk.DISABLED)
        set_global_log_widget(self.log_text)
        self.sample_frame = ttk.Frame(self.content_frame)
        self.sample_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self._populate_mode_buttons()
        self._update_mode_states()
        self.load_last_folder()

    def _build_header(self):
        self.header = ttk.Frame(self.content_frame)
        self.header.pack(fill=tk.X, pady=(0, 20))

        self.choose_folder_button = ttk.Button(
            self.header,
            text="Choose reference folder",
            command=self.choose_reference_folder,
        )
        self.choose_folder_button.grid(row=0, column=0, padx=(0, 20), rowspan=2, sticky="nsew")

        self.folder_label = ttk.Label(
            self.header,
            text="Reference folder:\nNone selected",
            font=("Helvetica", 12, "bold"),
            justify=tk.LEFT,
        )
        self.folder_label.grid(row=0, column=1, sticky="w")

        self.folder_hint_label = ttk.Label(
            self.header,
            text="",
            justify=tk.LEFT,
            wraplength=480,
        )
        self.folder_hint_label.grid(row=1, column=1, sticky="w")
        self.header.columnconfigure(1, weight=1)

        control_bar = ttk.Frame(self.header)
        control_bar.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(16, 0))
        ttk.Button(control_bar, text="Clear all",
                   command=lambda: confirm_clear_all(self)).pack(fill=tk.X, pady=(0, 10))
        ttk.Button(control_bar, text="Quit", command=self.safe_quit).pack(fill=tk.X)

    def _build_mode_buttons(self):
        self.mode_frame = ttk.Frame(self.content_frame)
        self.mode_frame.pack(fill=tk.X, pady=(0, 15))
        self.mode_frame.columnconfigure(0, weight=1)

    def _populate_mode_buttons(self):
        for child in self.mode_frame.winfo_children():
            child.destroy()
        for idx, (label, *_rest) in enumerate(self.mode_specs):
            btn_frame = tk.Frame(self.mode_frame, bd=0, highlightthickness=0, bg=self.cget("bg"))
            btn_frame.pack(side=tk.LEFT, padx=8, pady=5, expand=True)
            btn = ttk.Button(
                btn_frame,
                text=label,
                width=12,
                command=lambda l=label: self._handle_mode_click(l),
            )
            btn.pack(ipadx=12, ipady=8)
            self.mode_buttons[label] = {
                "button": btn,
                "subfolder": self.mode_specs[idx][1],
                "handler": self.mode_specs[idx][2],
            }

    def _build_utility_buttons(self):
        util_frame = ttk.Frame(self.content_frame)
        util_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Button(
            util_frame,
            text="Write data to template (DG only)",
            command=self.open_write_template_window,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))

        ttk.Button(
            util_frame,
            text="Batch processing",
            command=self.open_batch_window,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

    def _update_mode_states(self):
        for label, meta in self.mode_buttons.items():
            btn = meta["button"]
            subfolder = meta["subfolder"]
            handler = meta["handler"]
            if handler is None or subfolder is None:
                btn.state(["disabled"])
                continue
            if not self.current_folder:
                btn.state(["disabled"])
                continue
            target = os.path.join(self.current_folder, subfolder)
            if os.path.isdir(target):
                btn.state(["!disabled"])
            else:
                btn.state(["disabled"])

    def _handle_mode_click(self, label):
        meta = self.mode_buttons.get(label)
        if not meta:
            return
        handler = meta["handler"]
        subfolder = meta["subfolder"]
        if handler is None or subfolder is None:
            self._coming_soon(label)
            return
        if not self.current_folder:
            messagebox.showinfo("MuDRaW", "Please choose a reference folder first.")
            return
        target = os.path.join(self.current_folder, subfolder)
        if not os.path.isdir(target):
            messagebox.showwarning("MuDRaW", f"Folder '{subfolder}' not found inside the reference folder.")
            return
        handler(target, subfolder)

    def _coming_soon(self, label):
        messagebox.showinfo("MuDRaW", f"{label} processing is coming soon. Hopefully!")

    def load_last_folder(self):
        last_folder = load_last_folder_internal()
        if last_folder:
            last_folder = _abs_path(last_folder)
            display_name = _folder_name(last_folder)
            self.folder_label.config(text=f"Reference folder:\n{display_name}")
            self.folder_hint_label.config(text=last_folder)
            self.scan_subfolders(last_folder)

    def choose_reference_folder(self):
        initial_dir = _abs_path(load_last_folder_internal())
        try:
            folder_selected = filedialog.askdirectory(initialdir=initial_dir, title="Select Reference Folder")
        except Exception as e:
            messagebox.showerror("Error selecting folder", f"Could not open folder selection dialog:\n{e}")
            folder_selected = ""
        if folder_selected:
            folder_selected = _abs_path(folder_selected)
            display_name = _folder_name(folder_selected)
            self.folder_label.config(text=f"Reference folder:\n{display_name}")
            self.folder_hint_label.config(text=folder_selected)
            save_last_folder(folder_selected)
            self.scan_subfolders(folder_selected)

    def scan_subfolders(self, folder):
        folder = _abs_path(folder)
        # Remember the selected reference folder
        self.current_folder = folder

        # Ensure unified _Processed under the reference folder and index outputs
        self.processed_root = ensure_processed_root(folder)
        print("[MAIN] processed_root:", self.processed_root)
        self.available_outputs = index_output_jsons(self.processed_root)

        display_name = _folder_name(folder)
        self.folder_label.config(text=f"Reference folder:\n{display_name}")
        self.folder_hint_label.config(text=folder)

        self._update_mode_states()

    def open_batch_window(self):
        """Launch the BatchProcessingWindow Toplevel."""
        BW, err = _load_batch_window_class()
        if not BW:
            detail = f"\nDetails: {err}" if err else ""
            messagebox.showerror(
                "Batch Processing",
                "BatchProcessingWindow.py not found or failed to import.\n"
                "Make sure it sits next to main_window.py." + detail
            )
            return
        try:
            BW(self)
        except Exception as e:
            messagebox.showerror("Batch Processing", f"Failed to open window:\n{e}")

    def get_next_window_position(self, width=600, height=400, padding=30):
        """Find a non-overlapping position for a new window using a simple grid.
        Stores rectangles (x, y, w, h) into self.open_windows.
        """
        def _overlaps(x, y, w, h):
            for (ox, oy, ow, oh) in self.open_windows:
                if (x < ox + ow) and (x + w > ox) and (y < oy + oh) and (y + h > oy):
                    return True
            return False

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Margins to avoid menu bar/dock
        margin_x, margin_y = 40, 80
        step_x = max(width + padding, 100)
        step_y = max(height + padding, 100)
        cols = max(1, (screen_width - margin_x) // step_x)
        rows = max(1, (screen_height - margin_y) // step_y)

        # Try a bounded number of slots on the grid
        max_slots = max(20, cols * rows * 2)
        base_index = len(self.open_windows)
        for i in range(max_slots):
            r = (base_index + i) // max(cols, 1)
            c = (base_index + i) % max(cols, 1)
            x = margin_x + c * step_x
            y = margin_y + r * step_y
            if x + width >= screen_width or y + height >= screen_height:
                continue
            if not _overlaps(x, y, width, height):
                self.open_windows.append((x, y, width, height))
                return x, y
        # Fallback
        x = min(margin_x, screen_width - width)
        y = min(margin_y, screen_height - height)
        self.open_windows.append((x, y, width, height))
        return x, y

    def register_window_close(self, window, rect, extra_callback=None):
        def on_close():
            if extra_callback:
                try:
                    extra_callback()
                except Exception as exc:
                    print(f"[MAIN] window close hook error: {exc}")
            if rect in self.open_windows:
                try:
                    self.open_windows.remove(rect)
                except ValueError:
                    pass
            window.destroy()
        window.protocol("WM_DELETE_WINDOW", on_close)

    def open_rheology_folder(self, path, name):
        path = _clean_user_path(path)
        new_window = tk.Toplevel(self)
        new_window.title(f"Process {name} folder")
        x, y = self.get_next_window_position(600, 400)
        new_window.geometry(f"600x400+{x}+{y}")
        layout = ttk.Frame(new_window)
        layout.pack(fill=tk.BOTH, expand=True)

        reference_folder = getattr(self, "current_folder", os.path.dirname(path))
        reference_folder = _clean_user_path(reference_folder)

        mode_var = tk.StringVar(value="triggered")
        steady_var = tk.DoubleVar(value=10.0)

        params_frame = ttk.LabelFrame(layout, text="Rheology parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(params_frame, text="Mode").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        mode_cb = ttk.Combobox(
            params_frame,
            state="readonly",
            width=14,
            values=["triggered", "nontriggered", "other"],
            textvariable=mode_var,
        )
        mode_cb.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(params_frame, text="Steady window (s)").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        steady_entry = ttk.Entry(params_frame, width=10, textvariable=steady_var)
        steady_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        def _update_steady(*_):
            if mode_var.get() == "triggered":
                steady_entry.state(["disabled"])
            else:
                steady_entry.state(["!disabled"])

        mode_var.trace_add("write", _update_steady)
        _update_steady()

        ttk.Label(layout, text=f"Processing {name} folder: {path}").pack(pady=10)

        # List only CSV files, skip hidden and system files; natural sort
        def _list_csvs():
            exts = (".csv", ".txt", ".tsv")
            items = [f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))
                     and os.path.splitext(f)[1].lower() in exts
                     and not f.startswith('_')
                     and f != '.DS_Store']
            items.sort(key=_natural_key)
            return items

        samples_frame = ttk.Frame(new_window)
        samples_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        sample_files = _list_csvs()
        if not sample_files:
            ttk.Label(samples_frame, text="(No CSV files found)").pack(pady=6)

        def _run_rheo(csv_path):
            try:
                data = {"mode": mode_var.get(), "steady_sec": float(steady_var.get())}
                _rheo_save_session(reference_folder, data)
                process_rheology_file(
                    csv_path,
                    mode=mode_var.get(),
                    steady_sec=steady_var.get(),
                )
            except Exception as exc:
                messagebox.showerror("Rheology", f"Failed to launch reader:\n{exc}")

        for sample in sample_files:
            sample_name = os.path.splitext(sample)[0]
            full_path = os.path.join(path, sample)
            btn = ttk.Button(samples_frame, text=sample_name, command=lambda p=full_path: _run_rheo(p))
            btn.pack(fill=tk.X, pady=2)

        def _process_all():
            lst = _list_csvs()
            if not lst:
                messagebox.showinfo("Rheology", "No CSV files to process.")
                return
            for s in lst:
                _run_rheo(os.path.join(path, s))
        ttk.Button(new_window, text="Process all Rheology data", command=_process_all, width=24).pack(pady=5, fill=tk.X)

    def open_saxs_folder(self, path, name):
        path = _clean_user_path(path)
        new_window = tk.Toplevel(self)
        new_window.title(f"Process {name} folder")
        x, y = self.get_next_window_position(600, 400)
        new_window.geometry(f"600x400+{x}+{y}")
        self.register_window_close(new_window, (x, y, 600, 400))

        layout = ttk.Frame(new_window)
        layout.pack(fill=tk.BOTH, expand=True)

        smoo_default, sigma_default, lower_default, upper_default = get_last_used_saxs_parameters()
        smoo_var = tk.DoubleVar(value=smoo_default)
        sigma_var = tk.DoubleVar(value=sigma_default)
        lower_var = tk.DoubleVar(value=lower_default)
        upper_var = tk.DoubleVar(value=upper_default)
        fast_var = tk.BooleanVar(value=False)

        params_frame = ttk.LabelFrame(layout, text="SAXS parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=(10, 8))

        ttk.Label(params_frame, text="Smoothing").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(params_frame, width=8, textvariable=smoo_var).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(params_frame, text="Sigma").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(params_frame, width=8, textvariable=sigma_var).grid(row=0, column=3, sticky="w", padx=4, pady=2)
        ttk.Label(params_frame, text="θ min").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(params_frame, width=8, textvariable=lower_var).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(params_frame, text="θ max").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        ttk.Entry(params_frame, width=8, textvariable=upper_var).grid(row=1, column=3, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(params_frame, text="Fast (no plots)", variable=fast_var).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(4, 2))

        ttk.Label(layout, text=f"Processing {name} folder: {path}").pack(pady=8)

        sample_folders = [
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
            and not f.endswith('.dgraph')
            and not f.startswith('_')
        ]
        renamed_samples = load_renamed_samples()
        folder_refs = []

        def _run_single(ref, name_var):
            folder = ref["folder"]
            sample_label = name_var.get().strip() or folder
            sample_path = os.path.join(path, folder)
            try:
                engage_saxs(
                    sample_path,
                    sample_label,
                    self,
                    smoo_var.get(),
                    sigma_var.get(),
                    lower_var.get(),
                    upper_var.get(),
                    add_action,
                    update_actions_list,
                    fast_var=fast_var.get(),
                )
            except Exception as exc:
                messagebox.showerror("SAXS", f"Failed to process {sample_label}:\n{exc}")

        for sample in sample_folders:
            ref = {"folder": sample}
            folder_refs.append(ref)
            display_name = renamed_samples.get(sample, sample)
            name_var = tk.StringVar(value=display_name)

            row = ttk.Frame(layout)
            row.pack(fill=tk.X, padx=10, pady=2)

            btn = ttk.Button(row, textvariable=name_var,
                              command=lambda r=ref, nv=name_var: _run_single(r, nv))
            btn.pack(side=tk.LEFT)

            entry = ttk.Entry(row, textvariable=name_var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

            def _rename_current(ref=ref, var=name_var, entry_widget=entry):
                old_folder = ref["folder"]
                rename_sample(entry_widget, old_folder, path)
                new_name = var.get().strip() or old_folder
                ref["folder"] = new_name

            ttk.Button(row, text="Rename", command=_rename_current).pack(side=tk.LEFT)


    def open_pli_folder(self, path, name):
        path = _clean_user_path(path)
        new_window = tk.Toplevel(self)
        new_window.title(f"Process {name} folder")
        # Preferred compact size
        WIN_W, WIN_H = 350, 650
        x, y = self.get_next_window_position(WIN_W, WIN_H)
        new_window.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        new_window.update_idletasks()
        new_window.minsize(WIN_W, WIN_H)
        new_window.resizable(True, True)
        self.register_window_close(new_window, (x, y, WIN_W, WIN_H))

        label = ttk.Label(new_window, text=f"Processing {name} folder")
        label.pack(pady=10)

        video_exts = ('.mp4', '.avi', '.mov')

        video_files = [
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in video_exts and not f.startswith('_')
        ]
        video_files.sort(key=_natural_key)

        video_listbox = tk.Listbox(new_window, height=6, selectmode=tk.MULTIPLE)
        for f in video_files:
            video_listbox.insert(tk.END, f)
        video_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Insert the interactive extraction button directly under the video list
        from PLI_extract_st_diag_v2 import extract_space_time_interactive
        def interactive_extraction():
            try:
                selection = video_listbox.curselection()
                if not selection:
                    messagebox.showerror("Error", "No video selected for interactive extraction.")
                    return
                video_path = os.path.join(_clean_user_path(path), video_listbox.get(selection[0]))
                extract_space_time_interactive(video_path, "output")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to launch extractor:\n{e}")

        ttk.Button(new_window, text="Extract space-time diagram", command=interactive_extraction).pack(fill=tk.X, pady=(0, 8), padx=10)

        # Insert refresh button here (and its function)
        def refresh_image_list():
            files = list_raw_st_images(_clean_user_path(path))
            image_listbox.delete(0, tk.END)
            for f in sorted(files, key=_natural_key):
                image_listbox.insert(tk.END, f)
            if files:
                image_listbox.selection_set(0)
            messagebox.showinfo("Refresh complete", "Image list updated.")

        ttk.Button(new_window, text="Refresh image list", command=refresh_image_list).pack(fill=tk.X, pady=(5, 5), padx=10)

        image_files = list_raw_st_images(_clean_user_path(path))
        image_files.sort(key=_natural_key)

        ttk.Label(new_window, text="Detected image files (select up to two):").pack(pady=(10, 0))
        image_listbox = tk.Listbox(new_window, height=6, selectmode=tk.MULTIPLE, exportselection=False)
        for f in image_files:
            image_listbox.insert(tk.END, f)
        image_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        if image_files:
            image_listbox.selection_set(0)

        # Intervals and re-scaling button directly under image list
        def launch_intervals_rescaling():
            selection = image_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "No image selected for intervals and re-scaling.")
                return
            if len(selection) > 2:
                messagebox.showerror("Error", "Please select at most two images (they will be stitched horizontally).")
                return
            selected_images = [image_listbox.get(i) for i in selection]
            try:
                # Launch rescaler in a separate Python process to avoid Tk conflicts
                script_path = os.path.join(base_path, "PLI_rescale_space_time_diagram.py")
                if not os.path.isfile(script_path):
                    raise FileNotFoundError(f"Rescaling GUI script not found: {script_path}")
                workdir = _clean_user_path(path)
                # Pass the reference folder as the first CLI argument so the rescaler opens its dialogs there
                reference_folder = getattr(self, "current_folder", _clean_user_path(path))
                args = [sys.executable, script_path, reference_folder] + selected_images
                print("[MAIN] Launching PLI rescaler via subprocess:", args, "cwd=", workdir)
                subprocess.Popen(args, cwd=workdir)
            except Exception as e:
                messagebox.showerror(
                    "Launch error",
                    f"Failed to launch PLI rescaling interface.\n\nFolder: {_clean_user_path(path)}\nFiles: {selected_images}\n\nError: {e}"
                )

        ttk.Button(new_window, text="Intervals and re-scaling", command=launch_intervals_rescaling).pack(fill=tk.X, padx=10, pady=(0, 10))

        # --- Stitched outputs UI (prefer unscaled from PLI/_Temp) ---
        # Place the refresh button first (no command yet), so it sits above the list
        refresh_stitched_btn = ttk.Button(new_window, text="Refresh unscaled stitched (Temp)")
        refresh_stitched_btn.pack(fill=tk.X, padx=10, pady=(0, 6))

        ttk.Label(new_window, text="Unscaled stitched panels (PLI/_Temp):").pack(pady=(6, 0))

        stitched_listbox = tk.Listbox(new_window, height=6, selectmode=tk.SINGLE, exportselection=False)
        _stitched_items = list_unscaled_temp_stitched_outputs(_clean_user_path(path))
        _stitched_map = {os.path.basename(p): p for p in _stitched_items}
        for bp in sorted(_stitched_map.keys(), key=_natural_key):
            stitched_listbox.insert(tk.END, bp)
        stitched_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        def refresh_stitched_list():
            stitched_listbox.delete(0, tk.END)
            _items = list_unscaled_temp_stitched_outputs(_clean_user_path(path))
            _stitched_map.clear()
            _stitched_map.update({os.path.basename(p): p for p in _items})
            for bp in sorted(_stitched_map.keys(), key=_natural_key):
                stitched_listbox.insert(tk.END, bp)
            messagebox.showinfo("Refresh complete", "Unscaled stitched list (Temp) updated.")

        # Now that the function is defined and widgets exist, bind the command
        refresh_stitched_btn.config(command=refresh_stitched_list)

        # Button Frame
        button_frame = ttk.Frame(new_window)
        button_frame.pack(pady=10, fill=tk.X)

        def process_scaled_now():
            sel = stitched_listbox.curselection()
            source_path = None
            if sel:
                base = stitched_listbox.get(sel[0])
                stitched_full = _stitched_map.get(base)
                if not stitched_full or not stitched_full.lower().endswith(".png"):
                    messagebox.showerror("Error", "Please select a stitched color PNG (not a CSV/JSON).")
                    return
                low = stitched_full.lower()
                if ("_steady_unscaled_stitched_" not in low) and ("_stitched_" not in low):
                    messagebox.showerror("Error", "Please select a stitched color PNG (name contains '_stitched_').")
                    return
                source_path = stitched_full
            else:
                img_sel = image_listbox.curselection()
                if not img_sel:
                    messagebox.showerror("Error", "Select either an unscaled stitched PNG or a detected image first.")
                    return
                choice = image_listbox.get(img_sel[0])
                source_path = os.path.join(_clean_user_path(path), choice)
            try:
                open_maltese_cross_analyzer(source_path)
            except Exception as e:
                messagebox.showerror("Run error", f"Failed to open analyzer:\n{e}")

        ttk.Button(button_frame, text="Process data", command=process_scaled_now).pack(fill=tk.X, pady=2)

        def steady_state_extraction():
            selection = image_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "No image selected for steady-state extraction.")
                return
            selected_image = os.path.join(path, image_listbox.get(selection[0]))
            new_window = tk.Toplevel(self)
            new_window.title("Steady-state extraction")
            x, y = self.get_next_window_position(500, 150)
            new_window.geometry(f"500x150+{x}+{y}")
            self.register_window_close(new_window, (x, y))
            label = ttk.Label(new_window, text=f"Running steady-state extraction for {selected_image}")
            label.pack(pady=20)
            # Import subprocess and run the extractor script
            import subprocess
            subprocess.Popen([sys.executable, os.path.join(base_path, "_Helpers", "PLI_space_time_steady_state_extractor.py"), selected_image])
            new_window.lift()
            new_window.focus_force()

    def open_pi_folder(self, path, name):
        path = _clean_user_path(path)
        new_window = tk.Toplevel(self)
        new_window.title(f"Process {name} folder")
        WIN_W, WIN_H = 900, 640
        x, y = self.get_next_window_position(WIN_W, WIN_H)
        new_window.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        self.register_window_close(new_window, (x, y, WIN_W, WIN_H))
        new_window.minsize(780, 520)

        layout = ttk.Frame(new_window, padding=10)
        layout.pack(fill=tk.BOTH, expand=True)

        ttk.Label(layout, text=f"Processing {name} folder: {path}", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 8))

        details_frame = ttk.LabelFrame(layout, text="Sample options")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        details_container = ttk.Frame(details_frame)
        details_container.pack(fill=tk.BOTH, expand=True)
        placeholder = ttk.Label(details_container, text="Select a sample to view PI actions.", padding=20)
        placeholder.pack(expand=True)

        samples_frame = ttk.LabelFrame(layout, text="Detected samples")
        samples_frame.pack(fill=tk.X)

        sample_folders = [
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
            and not f.endswith('.dgraph')
            and not f.startswith('_')
        ]
        sample_folders.sort(key=_natural_key)
        renamed_samples = load_renamed_samples()

        def _select_sample(state):
            folder_name = state["folder"]
            display = state["name_var"].get().strip() or folder_name
            sample_root = os.path.join(path, folder_name)
            if not os.path.isdir(sample_root):
                messagebox.showerror("PI", f"Sample folder not found:\n{sample_root}")
                return
            working_folder = find_deepest_subfolder(sample_root)
            for child in details_container.winfo_children():
                child.destroy()
            process_sample_PI(
                self,
                working_folder,
                name,
                display,
                add_action,
                update_actions_list,
                lambda: self.current_folder or load_last_folder_internal(),
                target_container=details_container,
            )

        def _rename_state(state, entry_widget):
            old_folder = state["folder"]
            rename_sample(entry_widget, old_folder, path)
            new_name = entry_widget.get().strip() or old_folder
            state["folder"] = new_name

        if not sample_folders:
            ttk.Label(samples_frame, text="(No samples discovered)").pack(pady=6)
        else:
            for sample in sample_folders:
                state = {"folder": sample}
                name_var = tk.StringVar(value=renamed_samples.get(sample, sample))
                state["name_var"] = name_var

                row = ttk.Frame(samples_frame)
                row.pack(fill=tk.X, pady=2)

                ttk.Button(
                    row,
                    textvariable=name_var,
                    width=18,
                    command=lambda s=state: _select_sample(s),
                ).pack(side=tk.LEFT)

                entry = ttk.Entry(row, textvariable=name_var)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

                ttk.Button(
                    row,
                    text="Rename",
                    command=lambda s=state, e=entry: _rename_state(s, e),
                ).pack(side=tk.LEFT)

    def open_write_template_window(self):
        # Make the indexed outputs discoverable by the writer window via the parent (self)
        # Consumers can access: self.available_outputs and self.processed_root
        WriteTemplateWindow(self, base_path)

    def safe_quit(self):
        # Destroy all Toplevel windows and clean up
        for window in self.winfo_children():
            if isinstance(window, tk.Toplevel):
                try:
                    window.destroy()
                except Exception:
                    pass
        # Clear tracked open window positions
        self.open_windows.clear()
        self.destroy()

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
