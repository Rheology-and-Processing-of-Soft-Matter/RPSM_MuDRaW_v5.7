import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import numpy as np
import PLI_engine as engine

import sys, traceback

# Make Tk print exceptions instead of silently swallowing them
def _tk_report_exc(exc, val, tb):
    traceback.print_exception(exc, val, tb)
try:
    tk.Tk.report_callback_exception = staticmethod(_tk_report_exc)
except Exception:
    pass

# ... rest of the code ...

def browse_time_source(self):
    # Choose reference folder if available; fallback to last path or home
    try:
        if hasattr(self, 'reference_folder') and os.path.isdir(self.reference_folder):
            init_dir = self.reference_folder
        elif getattr(self, '_last_time_source_path', None) and os.path.isdir(self._last_time_source_path):
            init_dir = self._last_time_source_path
        else:
            init_dir = os.path.expanduser('~')
    except Exception:
        init_dir = os.path.expanduser('~')
    print("[UI] opening file dialog in reference folder… initialdir=", init_dir)
    path = filedialog.askopenfilename(
        parent=self,
        initialdir=init_dir,
        title="Select time source CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not path:
        return
    orig_path = path
    self._last_time_source_path = path
    # Let the user inspect and choose the time column explicitly (once)
    try:
        insp = TimeSourceInspector(self, path)
        self.wait_window(insp)
    except Exception as _e:
        print("[DBG] Inspector not shown:", _e)

    try:
        rows, prefer_flag, _delim = engine.read_csv_rows(path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV file: {e}\n{e.__class__.__name__}")
        return

    # Auto-convert classic Anton Paar TSV (UTF-16LE + tab-delimited) to a clean UTF-8 CSV
    if _delim == "\n":  # guard against mis-detection
        pass
    if _delim == "\t":
        try:
            clean_path = engine.auto_convert_anton_paar(path, header_skip=8)
            if clean_path and clean_path != path:
                # switch to the converted file before proceeding
                path = clean_path
                self._last_time_source_path = path
                print(f"[DBG] Using converted time-source: {os.path.basename(path)}")
                # propagate any manual override from the original path
                try:
                    if hasattr(self, '_TIME_COL_OVERRIDE') and isinstance(getattr(self, '_TIME_COL_OVERRIDE'), dict):
                        if orig_path in self._TIME_COL_OVERRIDE and path not in self._TIME_COL_OVERRIDE:
                            self._TIME_COL_OVERRIDE[path] = self._TIME_COL_OVERRIDE[orig_path]
                            print("[DBG] propagated time_idx override →", self._TIME_COL_OVERRIDE[path])
                except Exception as _prop_e:
                    print("[DBG] override propagation warning:", _prop_e)
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to auto-convert Anton Paar file: {e}")

    # Ensure FPS is initialized
    if not hasattr(self, '_fps'):
        try:
            self._fps = float(self.fps_var.get()) if hasattr(self, 'fps_var') else 29.97
        except Exception:
            self._fps = 29.97
    fps = self._fps
    # Compute advances (ignored here, but validates the time-column choice); correct API name is compute_pair_advances_from_times
    try:
        _adv, _bounds = engine.compute_pair_advances_from_times(path, fps, override=getattr(self, '_TIME_COL_OVERRIDE', {}))
    except Exception as e:
        print("[DBG] advances calc failed:", e)
    # Populate intervals based on current mode
    try:
        mode_sel = self.acq_mode.get()
        mode = (
            "Triggered" if mode_sel.startswith("Triggered") else
            ("Reference" if "reference" in mode_sel.lower() else "Manual")
        )
        self._compute_and_populate_from_time_source(path, mode)
    except Exception as e:
        print("[DBG] populate intervals failed:", e)

def refresh_intervals(self):
    path = self._last_time_source_path
    if not path or not os.path.isfile(path):
        print("[DBG] refresh skipped: no valid time-source file yet")
        return
    fps = self._fps
    try:
        _adv, _bounds = engine.compute_pair_advances_from_times(path, fps, override=getattr(self, '_TIME_COL_OVERRIDE', {}))
    except Exception as e:
        print("[DBG] advances calc failed (refresh):", e)
    # Rebuild the intervals table according to the current mode
    try:
        mode_sel = self.acq_mode.get()
        mode = (
            "Triggered" if mode_sel.startswith("Triggered") else
            ("Reference" if "reference" in mode_sel.lower() else "Manual")
        )
        self._compute_and_populate_from_time_source(path, mode)
    except Exception as e:
        print("[DBG] populate intervals failed (refresh):", e)

def _compute_and_populate_from_time_source(self, path, mode):
    """Populate the intervals table for the selected mode using the full set of intervals."""
    # Clear existing table contents
    for w in self.interval_frame.winfo_children():
        try:
            w.destroy()
        except Exception:
            pass

    # Column headers depend on mode
    if mode == "Triggered":
        # Fetch full Triggered pairs
        T_beg, T_end, S_beg, S_end = engine.extract_triggered_pairs_from_time_column(
            path, override=getattr(self, '_TIME_COL_OVERRIDE', {})
        )
        n = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
        # Headers
        headers = ["#", "T_begin [s]", "T_end [s]", "S_begin [s]", "S_end [s]", "T [frames]", "S [frames]"]
        for j, h in enumerate(headers):
            ttk.Label(self.interval_frame, text=h).grid(row=0, column=j, padx=6, pady=4, sticky="w")
        # Rows
        fps = getattr(self, '_fps', 29.97)
        for i in range(n):
            tb = float(T_beg[i]); te = float(T_end[i]); sb = float(S_beg[i]); se = float(S_end[i])
            T_frames = int(round(max(0.0, te - tb) * float(fps)))
            S_frames = int(round(max(0.0, se - sb) * float(fps)))
            ttk.Label(self.interval_frame, text=f"{i+1}").grid(row=i+1, column=0, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{tb:.3f}").grid(row=i+1, column=1, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{te:.3f}").grid(row=i+1, column=2, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{sb:.3f}").grid(row=i+1, column=3, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{se:.3f}").grid(row=i+1, column=4, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{T_frames}").grid(row=i+1, column=5, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{S_frames}").grid(row=i+1, column=6, padx=6, sticky="w")
        # Footer if no rows
        if n == 0:
            ttk.Label(self.interval_frame, text="(No Triggered intervals detected)").grid(row=1, column=0, padx=6, pady=6, sticky="w")

    elif mode == "Reference":
        # Reference mode: get segment begin/end, then derive T/S by subtracting steady_sec
        b_ref, e_ref = engine.parse_reference_steps_from_csv(
            path, override=getattr(self, '_TIME_COL_OVERRIDE', {})
        )
        try:
            steady_sec = float(self.steady_sec_var.get())
        except Exception:
            steady_sec = 10.0
        print(f"[UI] Reference steady_sec = {steady_sec} s")
        T_beg, T_end, S_beg, S_end = engine.compute_reference_intervals_with_steady(
            b_ref, e_ref, steady_sec
        )
        n = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
        # Headers like Triggered
        headers = ["#", "T_begin [s]", "T_end [s]", "S_begin [s]", "S_end [s]", "T [frames]", "S [frames]"]
        for j, h in enumerate(headers):
            ttk.Label(self.interval_frame, text=h).grid(row=0, column=j, padx=6, pady=4, sticky="w")
        fps = float(getattr(self, '_fps', 29.97))
        for i in range(n):
            tb = float(T_beg[i]); te = float(T_end[i]); sb = float(S_beg[i]); se = float(S_end[i])
            T_frames = int(round(max(0.0, te - tb) * fps))
            S_frames = int(round(max(0.0, se - sb) * fps))
            ttk.Label(self.interval_frame, text=f"{i+1}").grid(row=i+1, column=0, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{tb:.3f}").grid(row=i+1, column=1, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{te:.3f}").grid(row=i+1, column=2, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{sb:.3f}").grid(row=i+1, column=3, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{se:.3f}").grid(row=i+1, column=4, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{T_frames}").grid(row=i+1, column=5, padx=6, sticky="w")
            ttk.Label(self.interval_frame, text=f"{S_frames}").grid(row=i+1, column=6, padx=6, sticky="w")
        if n == 0:
            ttk.Label(self.interval_frame, text="(No Reference intervals detected)").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        # Cache these computed arrays on self so ENGAGE can reuse them if needed
        self._ref_T_beg = T_beg; self._ref_T_end = T_end; self._ref_S_beg = S_beg; self._ref_S_end = S_end

    else:
        # Manual mode: keep a simple placeholder for now
        headers = ["#", "Begin [s]", "End [s]"]
        for j, h in enumerate(headers):
            ttk.Label(self.interval_frame, text=h).grid(row=0, column=j, padx=6, pady=4, sticky="w")
        ttk.Label(self.interval_frame, text="(Manual mode: enter intervals below)").grid(row=1, column=0, padx=6, pady=6, sticky="w")

class PreviewEndWindow(tk.Toplevel):
    def __init__(self, parent, image_path, init_percent=5.0):
        super().__init__(parent)
        self.title("Preview end region")
        self.parent = parent
        self.image_path = image_path
        self.percent_var = tk.DoubleVar(value=float(init_percent))
        self._ax = None
        self._canvas = None
        self._end_x = 0
        self._line = None

        # Top controls
        top = ttk.Frame(self); top.pack(fill=tk.X, padx=10, pady=(10,6))
        ttk.Label(top, text="Show last (%) :").pack(side=tk.LEFT)
        pct_entry = ttk.Entry(top, textvariable=self.percent_var, width=6)
        pct_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(top, text="Refresh", command=self.refresh_view).pack(side=tk.LEFT)
        ttk.Label(top, text="End x (px):").pack(side=tk.LEFT, padx=(16,4))
        self.end_var = tk.IntVar(value=0)
        end_entry = ttk.Entry(top, textvariable=self.end_var, width=8)
        end_entry.pack(side=tk.LEFT)

        # Image area
        body = ttk.Frame(self); body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        self.fig, self._ax = plt.subplots(figsize=(9,4))
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self._canvas = FigureCanvasTkAgg(self.fig, master=body)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Slider for end position
        srow = ttk.Frame(self); srow.pack(fill=tk.X, padx=10, pady=(0,10))
        self.slider = ttk.Scale(srow, from_=0, to=1000, orient=tk.HORIZONTAL, command=self._on_slider)
        self.slider.pack(fill=tk.X)

        # Buttons
        brow = ttk.Frame(self); brow.pack(fill=tk.X, padx=10, pady=(0,10))
        ttk.Button(brow, text="Accept", command=self._accept).pack(side=tk.RIGHT, padx=6)
        ttk.Button(brow, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

        self.refresh_view()

    def _load_image(self):
        try:
            import cv2
            bgr = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read image: {self.image_path}")
            rgb = bgr[:, :, ::-1]
        except Exception:
            # Fallback: try matplotlib imread (slower, supports PNG/JPG)
            import matplotlib.image as mpimg
            rgb = mpimg.imread(self.image_path)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[...,None], 3, axis=2)
        return rgb

    def refresh_view(self):
        rgb = self._load_image()
        H, W = rgb.shape[:2]
        try:
            pct = float(self.percent_var.get() or 5.0)
        except Exception:
            pct = 5.0
        pct = max(1.0, min(100.0, pct))
        w0 = int(max(1, round(W * (1.0 - pct/100.0))))
        view = rgb[:, w0:]
        self._ax.clear()
        self._ax.imshow(view, aspect='auto')
        self._ax.set_title(f"End region: last {pct:.1f}%  (width={view.shape[1]} px)")
        self._ax.set_xlabel("x (px, cropped)")
        self._ax.set_ylabel("y (px)")
        # default end at right edge of cropped view
        self._end_x = view.shape[1] - 1
        try:
            self.end_var.set(int(self._end_x))
        except Exception:
            pass
        # slider maps to view width
        self.slider.configure(to=max(1, view.shape[1]-1))
        self._line = self._ax.axvline(self._end_x, color='red', linewidth=1.5)
        self._canvas.draw_idle()

    def _on_slider(self, *_):
        try:
            x = int(float(self.slider.get()))
        except Exception:
            x = 0
        self._end_x = x
        try:
            self.end_var.set(x)
        except Exception:
            pass
        if self._line is not None:
            self._line.set_xdata([x, x])
        self._canvas.draw_idle()

    def _accept(self):
        # store as absolute offset from the right edge of the full image
        try:
            rgb = self._load_image()
            W = rgb.shape[1]
        except Exception:
            W = None
        try:
            pct = float(self.percent_var.get() or 5.0)
        except Exception:
            pct = 5.0
        if W is not None:
            w0 = int(max(1, round(W * (1.0 - pct/100.0))))
            end_abs_x = w0 + int(self._end_x or 0)
            end_offset = max(0, W - end_abs_x)  # offset from right
            # keep on parent for later computations
            self.parent._end_offset_from_right = end_offset
            self.parent._end_abs_x = end_abs_x
            print(f"[UI] Accepted end x = {end_abs_x} (W={W}) → offset_from_right={end_offset}")
        self.destroy()


def preview_end_region(self):
    """Open the end-region preview for the current image (default last 5%)."""
    # Determine image to preview: prefer argv images (launched via subprocess), else prompt
    img_path = None
    try:
        if len(sys.argv) > 2:
            # prefer the first image arg that exists (relative to cwd if needed)
            for p in sys.argv[2:]:
                if os.path.isfile(p):
                    img_path = p; break
                rp = os.path.join(os.getcwd(), p)
                if os.path.isfile(rp):
                    img_path = rp; break
    except Exception:
        pass
    if not img_path:
        img_path = filedialog.askopenfilename(parent=self, title="Select space–time image",
                                              filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")])
    if not img_path:
        return
    try:
        win = PreviewEndWindow(self, img_path, init_percent=5.0)
        self.wait_window(win)
    except Exception as e:
        messagebox.showerror("Preview error", f"Failed to preview end region:\n{e}")


# --- ENGAGE action: Show full space–time image with cyan/red lines at intervals ---
def apply_crop_and_compute(self):
    """Render the full space–time image with cyan/red interval lines.
    Cyan = T_begin, Red = S_end. Uses current mode and the accepted end offset if available.
    """
    # Determine image to show: prefer argv images, else ask
    img_path = None
    try:
        if len(sys.argv) > 2:
            for p in sys.argv[2:]:
                if os.path.isfile(p):
                    img_path = p; break
                rp = os.path.join(os.getcwd(), p)
                if os.path.isfile(rp):
                    img_path = rp; break
    except Exception:
        pass
    if not img_path:
        img_path = filedialog.askopenfilename(parent=self, title="Select space–time image",
                                              filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")])
    if not img_path:
        return

    # Load image (BGR→RGB)
    try:
        import cv2
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = bgr[:, :, ::-1]
    except Exception:
        import matplotlib.image as mpimg
        rgb = mpimg.imread(img_path)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)

    H, W = rgb.shape[:2]

    # Get intervals from current mode/time-source
    path = getattr(self, '_last_time_source_path', None)
    if not path or not os.path.isfile(path):
        messagebox.showwarning("No time source", "Load a time source first.")
        return

    mode_sel = self.acq_mode.get()
    mode = ("Triggered" if mode_sel.startswith("Triggered") else ("Reference" if "reference" in mode_sel.lower() else "Manual"))

    if mode == "Triggered":
        T_beg, T_end, S_beg, S_end = engine.extract_triggered_pairs_from_time_column(
            path, override=getattr(self, '_TIME_COL_OVERRIDE', {})
        )
        n = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
        # Compute widths in frames
        fps = float(getattr(self, '_fps', 29.97))
        widths = []  # list of (wt, ws) per pair
        for i in range(n):
            wt = max(0.0, float(T_end[i]) - float(T_beg[i])) * fps
            ws = max(0.0, float(S_end[i]) - float(S_beg[i])) * fps
            widths.append((int(round(wt)), int(round(ws))))
    elif mode == "Reference":
        b_ref, e_ref = engine.parse_reference_steps_from_csv(
            path, override=getattr(self, '_TIME_COL_OVERRIDE', {})
        )
        try:
            steady_sec = float(self.steady_sec_var.get())
        except Exception:
            steady_sec = 10.0
        print(f"[UI] Reference steady_sec = {steady_sec} s")
        T_beg, T_end, S_beg, S_end = engine.compute_reference_intervals_with_steady(
            b_ref, e_ref, steady_sec
        )
        n = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
        fps = float(getattr(self, '_fps', 29.97))
        widths = []
        for i in range(n):
            wt = max(0.0, float(T_end[i]) - float(T_beg[i])) * fps
            ws = max(0.0, float(S_end[i]) - float(S_beg[i])) * fps
            widths.append((int(round(wt)), int(round(ws))))
    else:
        messagebox.showinfo("Manual mode", "Manual mode preview not implemented yet.")
        return

    # Determine anchor: end-of-last-interval x on the image (default right edge)
    end_offset = int(getattr(self, '_end_offset_from_right', 0) or 0)
    end_x = W - max(0, min(end_offset, W))

    # Build figure in a new window
    win = tk.Toplevel(self)
    win.title("Preview with intervals")
    fig, ax = plt.subplots(figsize=(min(12, W/120), min(6, H/200)))
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=win)
    ax.imshow(rgb, aspect='auto')
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")

    # Place lines using absolute seconds: x(t) = end_x - (t_last - t)*fps
    fps = float(getattr(self, '_fps', 29.97))
    if mode == "Triggered":
        if n == 0:
            messagebox.showinfo("No intervals", "No Triggered intervals detected.")
            return
        t_last = float(S_end[-1])  # last steady end in seconds
        # Cyan = T_begin, Red = S_end
        x_cyan = [int(round(end_x - (t_last - float(t)) * fps)) for t in T_beg]
        x_red  = [int(round(end_x - (t_last - float(t)) * fps)) for t in S_end]
    else:  # Reference
        if n == 0:
            messagebox.showinfo("No intervals", "No Reference intervals detected.")
            return
        t_last = float(S_end[-1])
        x_cyan = [int(round(end_x - (t_last - float(t)) * fps)) for t in T_beg]
        x_red  = [int(round(end_x - (t_last - float(t)) * fps)) for t in S_end]

    # Draw with clipping to image bounds
    for x0 in x_cyan:
        if 0 <= x0 < W:
            ax.axvline(x0, color='cyan', linewidth=1.0)
    for xr in x_red:
        if 0 <= xr < W:
            ax.axvline(xr, color='red', linewidth=1.0)

    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Accept & Stitch controls ---
    ctrl = ttk.Frame(win)
    ctrl.pack(fill=tk.X, padx=10, pady=(0,10))

    def _stitch_now():
        try:
            import cv2
            # target size from UI: Height + Aspect Ratio (W/H)
            try:
                target_h = int(self.scaled_height_var.get())
            except Exception:
                target_h = 300
            try:
                aspect = float(self.aspect_var.get())
            except Exception:
                aspect = 1.426
            target_h = max(1, target_h)
            aspect = max(0.1, aspect)  # prevent degenerate widths

            # Build x-start/x-end for steady tiles in chronological order
            fps_loc = float(getattr(self, '_fps', 29.97))
            if mode == "Triggered":
                t_last = float(S_end[-1])
                xs0 = [int(round(end_x - (t_last - float(s)) * fps_loc)) for s in S_beg]
                xs1 = [int(round(end_x - (t_last - float(s)) * fps_loc)) for s in S_end]
            else:  # Reference
                t_last = float(S_end[-1])
                xs0 = [int(round(end_x - (t_last - float(s)) * fps_loc)) for s in S_beg]
                xs1 = [int(round(end_x - (t_last - float(s)) * fps_loc)) for s in S_end]

            tiles = []          # scaled tiles (preview)
            full_tiles = []     # unscaled tiles (analysis)
            for x0, x1 in zip(xs0, xs1):
                x0c = max(0, min(W, x0))
                x1c = max(0, min(W, x1))
                if x1c <= x0c:
                    continue
                tile = rgb[:, x0c:x1c]
                full_tiles.append(tile)

            if not full_tiles:
                messagebox.showwarning("Nothing to stitch", "No steady-state tiles could be extracted.")
                return

            # --- Unscaled stitched (analysis) → <reference>/PLI/_Temp
            unscaled = np.hstack(full_tiles)
            try:
                base = getattr(self, 'reference_folder', None) or os.getcwd()
                temp_dir = os.path.join(base, "PLI", "_Temp")
                os.makedirs(temp_dir, exist_ok=True)
            except Exception:
                temp_dir = os.getcwd()
            base_img = os.path.splitext(os.path.basename(img_path))[0]
            unscaled_name = f"{base_img}_steady_unscaled_stitched_{unscaled.shape[1]}x{unscaled.shape[0]}.png"
            unscaled_path = os.path.join(temp_dir, unscaled_name)
            try:
                bgr_unscaled = unscaled[:, :, ::-1]
            except Exception:
                bgr_unscaled = unscaled
            if not cv2.imwrite(unscaled_path, bgr_unscaled):
                raise RuntimeError("Failed to write unscaled stitched image")
            print("[STITCH] Wrote unscaled:", unscaled_path)

            # --- Scaled stitched (preview) uses (height, aspect) ---
            n_tiles = len(full_tiles)
            target_total_w = int(round(aspect * target_h))
            min_total_w = max(n_tiles, 1)
            if target_total_w < min_total_w:
                print(f"[STITCH] Requested width {target_total_w}px < number of tiles {n_tiles}. Using {min_total_w}px instead.")
                target_total_w = min_total_w

            base_w = target_total_w // n_tiles
            rem = target_total_w - base_w * n_tiles
            tile_widths = [base_w + (1 if i < rem else 0) for i in range(n_tiles)]

            for tile, w_i in zip(full_tiles, tile_widths):
                w_i = max(1, int(w_i))
                interp = cv2.INTER_AREA if (tile.shape[0] > target_h or tile.shape[1] > w_i) else cv2.INTER_LINEAR
                tiles.append(cv2.resize(tile, (w_i, target_h), interpolation=interp))

            stitched = np.hstack(tiles)
            if stitched.shape[0] != target_h:
                stitched = cv2.resize(stitched, (stitched.shape[1], target_h), interpolation=cv2.INTER_NEAREST)

            try:
                out_dir = os.path.join(base, "_Processed", "PLI")
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                out_dir = os.getcwd()
            out_name = f"{base_img}_steady_rescaled_stitched_{stitched.shape[1]}x{stitched.shape[0]}.png"
            out_path = os.path.join(out_dir, out_name)
            try:
                bgr_out = stitched[:, :, ::-1]
            except Exception:
                bgr_out = stitched
            if not cv2.imwrite(out_path, bgr_out):
                raise RuntimeError("Failed to write scaled stitched image")

            messagebox.showinfo(
                "Stitched",
                "Saved unscaled stitched (analysis) to:\n" + unscaled_path +
                "\n\nSaved scaled stitched (preview) to:\n" + out_path +
                "\n\n(Processing should read from PLI/_Temp; scaled image in _Processed is not used by Write to template.)"
            )
        except Exception as e:
            messagebox.showerror("Stitch error", f"Failed to stitch:\n{e}")

    ttk.Button(ctrl, text="Accept & Stitch", command=_stitch_now).pack(side=tk.RIGHT)
    ttk.Button(ctrl, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=(0,8))

class TimeSourceInspector(tk.Toplevel):
    def __init__(self, parent, csv_path):
        super().__init__(parent)
        self.title("Inspect time source: choose Time column")
        self.geometry("900x500")
        self.parent = parent
        self.csv_path = csv_path

        try:
            rows, _pref, _delim = engine.read_csv_rows(csv_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file: {e}")
            self.destroy(); return

        # show first ~40 rows
        N = min(40, len(rows))
        width = max((len(r) for r in rows[:N]), default=0)

        frm = ttk.Frame(self); frm.pack(fill=tk.BOTH, expand=True)
        cols = [f"C{j}" for j in range(width)]
        tree = ttk.Treeview(frm, columns=cols, show='headings')
        for j in range(width):
            tree.heading(f"C{j}", text=f"Col {j}")
            tree.column(f"C{j}", width=100, anchor='w')
        for i in range(N):
            row = rows[i] if i < len(rows) else []
            vals = [str(row[j]) if j < len(row) else '' for j in range(width)]
            tree.insert('', 'end', values=vals)
        yscroll = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=yscroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # controls
        ctrl = ttk.Frame(self); ctrl.pack(fill=tk.X, pady=8)
        ttk.Label(ctrl, text="Time column index:").pack(side=tk.LEFT, padx=6)
        # Use StringVar so transient empty values during typing don't raise TclError
        self.col_var = tk.StringVar(value="1")
        spin = ttk.Spinbox(ctrl, from_=0, to=max(0, width-1), textvariable=self.col_var, width=6)
        spin.pack(side=tk.LEFT)
        self.feedback = ttk.Label(ctrl, text="")
        self.feedback.pack(side=tk.LEFT, padx=12)

        def update_feedback(*_):
            try:
                col_str = str(self.col_var.get()).strip()
            except Exception:
                col_str = ""
            try:
                col = int(col_str) if col_str != "" else 0
            except Exception:
                col = 0
            # Clamp column to valid range
            if col < 0:
                col = 0
            if width > 0 and col >= width:
                col = width - 1
            hits = 0
            for r in rows[:30]:
                s = r[col] if col < len(r) else ''
                v = engine.parse_float_cell(s, False)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    hits += 1
            self.feedback.configure(text=f"parsed hits in first 30 rows: {hits}")
        self.col_var.trace_add('write', update_feedback)
        update_feedback()

        btns = ttk.Frame(self); btns.pack(fill=tk.X)
        def on_ok():
            try:
                col = int(self.col_var.get())
            except Exception:
                col = 0
            # store override on the parent window
            try:
                if not hasattr(self.parent, '_TIME_COL_OVERRIDE'):
                    self.parent._TIME_COL_OVERRIDE = {}
                self.parent._TIME_COL_OVERRIDE[self.csv_path] = col
                print("[DBG] time_idx override set →", col)
            except Exception as _e:
                print("[DBG] failed to set override:", _e)
            self.destroy()
        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)

import json

class RescaleSpaceTimeWindow(tk.Toplevel):
    def __init__(self, *args, **kwargs):
        print("[UI] __init__ enter")
        super().__init__(*args, **kwargs)
        # Per-window manual override map (path -> chosen time column index)
        self._TIME_COL_OVERRIDE = {}
        # Per-dataset session persistence
        self._session_state = {}
        self._session_file = os.path.expanduser("~/.rpsm_rescale_st_session.json")
        # Ensure window has a proper title and initial size
        try:
            self.title("Rescale Space–Time Diagram")
            # Only set a default geometry if none provided
            if not self.winfo_width() or not self.winfo_height():
                self.geometry("1100x820")
        except Exception:
            pass
        # ---- Basic UI skeleton ----
        root_frame = ttk.Frame(self)
        root_frame.pack(fill=tk.BOTH, expand=True)

        # Top controls
        top = ttk.LabelFrame(root_frame, text="Controls")
        top.pack(fill=tk.X, padx=10, pady=(10, 6))

        # FPS
        self.fps_var = tk.DoubleVar(value=float(getattr(self, "_fps", 29.97)))
        fps_entry = ttk.Entry(top, textvariable=self.fps_var, width=10)
        fps_entry.grid(row=0, column=1, sticky="w")
        def _sync_fps(*_):
            try:
                self._fps = float(self.fps_var.get())
            except Exception:
                self._fps = 29.97
        self.fps_var.trace_add("write", _sync_fps)
        # Initialize backing attribute so early callers see a value
        try:
            self._fps = float(self.fps_var.get())
        except Exception:
            self._fps = 29.97

        # Scaled preview sizing: Height + Aspect Ratio (W/H)
        ttk.Label(top, text="Scaled height (px):").grid(row=0, column=2, sticky="w", padx=(20,6))
        self.scaled_height_var = tk.IntVar(value=350)
        ttk.Entry(top, textvariable=self.scaled_height_var, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(top, text="Aspect ratio W/H:").grid(row=0, column=4, sticky="w", padx=(20,6))
        self.aspect_var = tk.DoubleVar(value=1.426)  # e.g., width = 6 × height
        ttk.Entry(top, textvariable=self.aspect_var, width=8).grid(row=0, column=5, sticky="w")

        # Mode radio buttons
        self.acq_mode = tk.StringVar(value="Triggered")
        modes = ttk.LabelFrame(root_frame, text="Steady-state interval definition mode:")
        modes.pack(fill=tk.X, padx=10, pady=(0,6))
        ttk.Radiobutton(modes, text="Triggered steady-state intervals", variable=self.acq_mode, value="Triggered").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(modes, text="Steady-state intervals not triggered — reference rheology file", variable=self.acq_mode, value="Reference").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Radiobutton(modes, text="Steady-state intervals not triggered — manual entry", variable=self.acq_mode, value="Manual").grid(row=2, column=0, sticky="w", padx=6, pady=2)

        # File controls
        file_row = ttk.Frame(root_frame)
        file_row.pack(fill=tk.X, padx=10, pady=(0,6))
        ttk.Button(file_row, text="Load time source…", command=self.browse_time_source).pack(side=tk.LEFT)
        ttk.Button(file_row, text="↻ Refresh intervals", command=self.refresh_intervals).pack(side=tk.LEFT, padx=(8,0))

        # Reference-mode steady width (seconds), appears only when Reference is selected
        self._ref_opts = ttk.Frame(root_frame)
        self._ref_opts.pack(fill=tk.X, padx=10, pady=(0,6))
        ttk.Label(self._ref_opts, text="Steady state width (s):").pack(side=tk.LEFT)
        self.steady_sec_var = tk.DoubleVar(value=10.0)
        ttk.Entry(self._ref_opts, textvariable=self.steady_sec_var, width=8).pack(side=tk.LEFT, padx=(6,0))

        def _toggle_ref_opts(*_):
            try:
                mode_now = str(self.acq_mode.get())
            except Exception:
                mode_now = "Triggered"
            if mode_now == "Reference":
                try:
                    self._ref_opts.pack(fill=tk.X, padx=10, pady=(0,6))
                except Exception:
                    pass
            else:
                try:
                    self._ref_opts.forget()
                except Exception:
                    pass
        self.acq_mode.trace_add('write', _toggle_ref_opts)
        _toggle_ref_opts()

        # Intervals table (scrollable)
        tbl_frame = ttk.LabelFrame(root_frame, text="Intervals")
        tbl_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        canvas = tk.Canvas(tbl_frame, highlightthickness=0)
        yscroll = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.interval_frame = ttk.Frame(canvas)
        self.interval_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.interval_frame, anchor="nw")
        canvas.configure(yscrollcommand=yscroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Seed with a tiny placeholder so the table isn't empty
        ttk.Label(self.interval_frame, text="(Intervals will appear here after loading time source)").grid(row=0, column=0, padx=6, pady=6, sticky="w")

        # Action buttons
        btn_row = ttk.Frame(root_frame)
        btn_row.pack(fill=tk.X, padx=10, pady=(0,10))
        ttk.Button(btn_row, text="PREVIEW", command=getattr(self, 'preview_end_region', lambda: None)).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="ENGAGE", command=getattr(self, 'apply_crop_and_compute', lambda: None)).pack(side=tk.LEFT, padx=(8,0))

        # Bring window to front on macOS and print readiness
        try:
            self.update_idletasks()
            self.deiconify()
            self.lift()
            self.focus_force()
        except Exception as _e:
            print("[DBG] UI focus hint failed:", _e)
        print("[UI] __init__ ready")
        self.after(200, lambda: print("[UI] event loop alive"))
        # Second-chance raise in case the root steals focus
        try:
            self.wait_visibility()
            self.lift()
            self.focus_force()
        except Exception:
            pass

        # Load session state on startup
        self._load_session()

    # --- Session persistence ---
    def _save_session(self):
        """Save per-dataset session info (intervals, selected column, end offset, mode) to disk."""
        try:
            # Session state is a dict: { dataset_path : { ... } }
            session = {}
            # Load previous state if file exists
            if os.path.isfile(self._session_file):
                with open(self._session_file, "r") as f:
                    try:
                        session = json.load(f)
                    except Exception:
                        session = {}
            # Save current for current dataset (if any)
            path = getattr(self, "_last_time_source_path", None)
            if path:
                d = {}
                # Save selected column
                col_map = getattr(self, "_TIME_COL_OVERRIDE", {})
                if path in col_map:
                    d["selected_col"] = col_map[path]
                # Save end offset and absolute x (if set)
                if hasattr(self, "_end_offset_from_right"):
                    d["end_offset_from_right"] = int(self._end_offset_from_right)
                if hasattr(self, "_end_abs_x"):
                    d["end_abs_x"] = int(self._end_abs_x)
                # Save mode
                try:
                    d["acq_mode"] = str(self.acq_mode.get())
                except Exception:
                    pass
                # Save sizing
                try:
                    d["scaled_height"] = int(self.scaled_height_var.get())
                except Exception:
                    pass
                try:
                    d["aspect_wh"] = float(self.aspect_var.get())
                except Exception:
                    pass
                session[path] = d
            # Write back
            with open(self._session_file, "w") as f:
                json.dump(session, f, indent=2)
        except Exception as e:
            print("[DBG] Failed to save session:", e)

    def _load_session(self):
        """Load session info for the last used dataset, if available."""
        try:
            if not os.path.isfile(self._session_file):
                return
            with open(self._session_file, "r") as f:
                session = json.load(f)
            # If last path is known, restore state for it
            path = getattr(self, "_last_time_source_path", None)
            if path and path in session:
                d = session[path]
                # Restore selected column
                if "selected_col" in d:
                    if not hasattr(self, "_TIME_COL_OVERRIDE"):
                        self._TIME_COL_OVERRIDE = {}
                    self._TIME_COL_OVERRIDE[path] = d["selected_col"]
                # Restore end offset/abs x
                if "end_offset_from_right" in d:
                    self._end_offset_from_right = d["end_offset_from_right"]
                if "end_abs_x" in d:
                    self._end_abs_x = d["end_abs_x"]
                # Restore mode
                if "acq_mode" in d:
                    try:
                        self.acq_mode.set(d["acq_mode"])
                    except Exception:
                        pass
                # Restore sizing
                if "scaled_height" in d:
                    try:
                        self.scaled_height_var.set(int(d["scaled_height"]))
                    except Exception:
                        pass
                if "aspect_wh" in d:
                    try:
                        self.aspect_var.set(float(d["aspect_wh"]))
                    except Exception:
                        pass
        except Exception as e:
            print("[DBG] Failed to load session:", e)

    # --- Shims: forward UI callbacks to module-level implementations ---
    def browse_time_source(self):
        result = browse_time_source(self)
        # After loading, save session for new dataset and reload previous session state
        self._save_session()
        self._load_session()
        return result

    def refresh_intervals(self):
        result = refresh_intervals(self)
        self._save_session()
        return result

    def _compute_and_populate_from_time_source(self, path, mode):
        result = _compute_and_populate_from_time_source(self, path, mode)
        self._save_session()
        return result

    def preview_end_region(self):
        result = preview_end_region(self)
        self._save_session()
        return result

    def apply_crop_and_compute(self):
        result = apply_crop_and_compute(self)
        self._save_session()
        return result

if __name__ == "__main__":
    # Standalone launch: create a real Tk root and host the Toplevel in it
    try:
        root = tk.Tk()
    except Exception as e:
        print("[ERR] Failed to create Tk root:", e)
        raise
    # Hide the root window (some WM need both withdraw & iconify to fully hide)
    try:
        root.withdraw()
        root.iconify()
    except Exception:
        pass
    app = RescaleSpaceTimeWindow(master=root)
    # If a reference folder is passed as argv[1], set it on the app
    try:
        if len(sys.argv) > 1:
            ref_cli = sys.argv[1]
            if os.path.isdir(ref_cli):
                app.reference_folder = ref_cli
                app._last_time_source_path = ref_cli
                print(f"[UI] Using reference folder from CLI → {ref_cli}")
    except Exception as _e:
        print("[DBG] failed to set reference folder from argv:", _e)
    # Load session on startup (again, after CLI path may be set)
    try:
        app._load_session()
    except Exception:
        pass
    # Make sure the Toplevel is raised and the root remains hidden
    try:
        root.withdraw()
        app.deiconify()
        app.lift()
        app.focus_force()
    except Exception as _e:
        print("[DBG] post-launch focus hint failed:", _e)
    try:
        app.mainloop()
    except Exception as e:
        import traceback
        print("[ERR] mainloop exited with error:", e)
        traceback.print_exc()