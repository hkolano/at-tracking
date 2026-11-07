"""Measure tag poses relative to a chosen reference tag.

Run from the repository root:
    python -m src.gui --config config/example.yaml
"""

import argparse
import math
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from src.capture import CaptureWorker
from src.config import load_config
from src.measurement import DISPLAY_FILTER_SECONDS, clean_id

POLL_MILLISECONDS = 30
DEFAULT_SECONDS = 5
COLOR_ERROR = "#b3261e"
COLOR_NORMAL = "#1f1f1f"


def slider_to_exposure(position, low, high):
    """Convert a slider position from 0 to 1 into an exposure on a log scale."""
    low = max(low, 1.0)
    return round(low * (high / low) ** position)


def exposure_to_slider(exposure, low, high):
    """Convert an exposure into a slider position from 0 to 1."""
    low = max(low, 1.0)
    position = math.log(max(exposure, low) / low) / math.log(high / low)
    return min(position, 1.0)


class TrackerApp:
    """Window showing the camera, live tag positions, and recording controls."""

    def __init__(self, root, worker):
        """Build the window and start polling the worker."""
        self.root = root
        self.worker = worker
        self.camera_failed = False
        self.last_sequence = 0
        self.photo = None  # Tk only shows an image while a reference to it exists.
        self.rows = {}     # tag_id -> table row
        self.exposure_ready = False

        self.root.title("Tag Measurement")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_camera_view()
        self._build_controls()
        self._build_status_bar()

        self.root.after(POLL_MILLISECONDS, self.poll)

    def _build_camera_view(self):
        """Create the canvas that shows the camera image."""
        self.canvas = tk.Canvas(self.root, width=960, height=540, background="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas_text = self.canvas.create_text(
            20, 20, anchor="nw", fill="white", text="Starting camera...", width=900
        )

    def _build_controls(self):
        """Create the reference selector, tag table, and recording controls."""
        panel = ttk.Frame(self.root, padding=10)
        panel.grid(row=0, column=1, sticky="ns")
        panel.rowconfigure(1, weight=1)

        reference_row = ttk.Frame(panel)
        reference_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(reference_row, text="Reference tag:").pack(side="left")
        self.reference_box = ttk.Combobox(reference_row, state="readonly", width=8)
        self.reference_box.pack(side="left", padx=(6, 0))
        self.reference_box.bind("<<ComboboxSelected>>", self._on_reference_selected)

        columns = ("tag", "name", "x", "y", "z")
        self.table = ttk.Treeview(panel, columns=columns, show="headings", height=14)
        headings = {"tag": "Tag", "name": "Name", "x": "x (m)", "y": "y (m)", "z": "z (m)"}
        widths = {"tag": 45, "name": 100, "x": 75, "y": 75, "z": 75}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="e" if column in "xyz" else "w")
        self.table.grid(row=1, column=0, sticky="ns")

        self.hint = ttk.Label(panel, text="Choose a reference tag.", wraplength=360)
        self.hint.grid(row=2, column=0, sticky="w", pady=(6, 10))

        exposure_row = ttk.Frame(panel)
        exposure_row.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(exposure_row, text="Exposure:").pack(side="left")
        self.exposure_scale = ttk.Scale(exposure_row, from_=0.0, to=1.0, state="disabled")
        self.exposure_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.exposure_label = ttk.Label(exposure_row, text="--", width=6)
        self.exposure_label.pack(side="left")
        self.auto_exposure = tk.BooleanVar(value=False)
        self.auto_check = ttk.Checkbutton(
            exposure_row,
            text="Auto",
            variable=self.auto_exposure,
            command=self._on_auto_exposure,
            state="disabled",
        )
        self.auto_check.pack(side="left")

        id_row = ttk.Frame(panel)
        id_row.grid(row=4, column=0, sticky="ew")
        ttk.Label(id_row, text="Bush ID:").pack(side="left")
        self.bush_entry = ttk.Entry(id_row, width=10)
        self.bush_entry.pack(side="left", padx=(6, 12))
        ttk.Label(id_row, text="Cane ID:").pack(side="left")
        self.cane_entry = ttk.Entry(id_row, width=10)
        self.cane_entry.pack(side="left", padx=(6, 0))

        record_row = ttk.Frame(panel)
        record_row.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(record_row, text="Seconds:").pack(side="left")
        self.seconds_box = ttk.Spinbox(record_row, from_=0.5, to=600, increment=1, width=6)
        self.seconds_box.set(DEFAULT_SECONDS)
        self.seconds_box.pack(side="left", padx=(6, 12))
        self.record_button = ttk.Button(record_row, text="Record", command=self._on_record, state="disabled")
        self.record_button.pack(side="right", fill="x", expand=True)

    def _build_status_bar(self):
        """Create the frame rate and message labels."""
        bar = ttk.Frame(self.root, padding=(10, 4))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.fps_label = ttk.Label(bar, text="-- fps", width=10)
        self.fps_label.pack(side="left")
        self.status_label = ttk.Label(bar, text="", wraplength=1100)
        self.status_label.pack(side="left", fill="x", expand=True)

    def _on_reference_selected(self, _event):
        """Send the chosen reference tag to the worker."""
        self.worker.set_reference(int(self.reference_box.get()))

    def _on_exposure_slider(self, position):
        """Send the slider's exposure to the worker."""
        low, high = self.worker.get_state().exposure_range
        exposure = slider_to_exposure(float(position), low, high)
        self.exposure_label.configure(text=str(exposure))
        self.worker.set_exposure(exposure)

    def _on_auto_exposure(self):
        """Send the Auto checkbox's value to the worker."""
        self.worker.set_auto_exposure(self.auto_exposure.get())

    def _on_record(self):
        """Validate the recording length and start a recording."""
        try:
            seconds = float(self.seconds_box.get())
            self.worker.start_recording(seconds, self.bush_entry.get(), self.cane_entry.get())
        except ValueError:
            self._set_status("Enter a recording length above 0 seconds.", is_error=True)
            return
        self.record_button.configure(state="disabled")
        self._set_status("Recording...")

    def _set_status(self, text, is_error=False):
        """Show a message in the status bar."""
        self.status_label.configure(text=text, foreground=COLOR_ERROR if is_error else COLOR_NORMAL)

    def poll(self):
        """Show the worker's latest results, then schedule the next poll."""
        self._show_notices()
        state = self.worker.get_state()
        if state is not None and state.sequence != self.last_sequence:
            self.last_sequence = state.sequence
            self._show_frame(state.frame)
            self._update_reference_choices(state)
            self._update_table(state)
            self._set_up_exposure(state)
            self.fps_label.configure(text=f"{state.fps:.1f} fps")
        self._update_controls(state)
        self.root.after(POLL_MILLISECONDS, self.poll)

    def _show_notices(self):
        """Show worker messages, joining those that arrive together."""
        notices = self.worker.get_notices()
        if not notices:
            return
        self.camera_failed = self.camera_failed or any(notice.is_fatal for notice in notices)
        has_error = any(notice.is_error for notice in notices)
        self._set_status("  ".join(notice.text for notice in notices), is_error=has_error)
        if self.camera_failed:
            self.canvas.itemconfigure(self.canvas_text, text="Camera unavailable.")

    def _show_frame(self, frame):
        """Scale a BGR frame to fit the canvas and display it."""
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        frame_height, frame_width = frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        size = (max(int(frame_width * scale), 1), max(int(frame_height * scale), 1))

        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.canvas_image, image=self.photo)
        self.canvas.itemconfigure(self.canvas_text, text="")

    def _update_reference_choices(self, state):
        """Offer every tag seen so far as a reference choice."""
        choices = tuple(str(tag_id) for tag_id in state.seen_ids)
        if state.reference_id is not None and state.reference_id not in state.seen_ids:
            choices = (str(state.reference_id), *choices)
        if tuple(self.reference_box.cget("values")) != choices:
            self.reference_box.configure(values=choices)
        if state.reference_id is not None and self.reference_box.get() != str(state.reference_id):
            self.reference_box.set(str(state.reference_id))

    def _update_table(self, state):
        """Show every non-reference tag, removing the row of a new reference."""
        tag_ids = [tag_id for tag_id in state.seen_ids if tag_id != state.reference_id]
        for tag_id in set(self.rows) - set(tag_ids):
            self.table.delete(self.rows.pop(tag_id))

        for tag_id in tag_ids:
            values = self._row_values(tag_id, state)
            if tag_id in self.rows:
                self.table.item(self.rows[tag_id], values=values)
            else:
                position = sum(other < tag_id for other in self.rows)
                self.rows[tag_id] = self.table.insert("", position, values=values)

    def _set_up_exposure(self, state):
        """Match the exposure controls to the camera on the first frame."""
        if self.exposure_ready:
            return
        self.exposure_ready = True
        low, high = state.exposure_range
        # Set before attaching the command, so this doesn't send a change.
        self.exposure_scale.configure(state="normal")
        self.exposure_scale.set(exposure_to_slider(state.exposure, low, high))
        self.exposure_scale.configure(command=self._on_exposure_slider)
        self.exposure_label.configure(text=str(round(state.exposure)))
        self.auto_exposure.set(state.auto_exposure)

    def _hint_text(self, state):
        """Return what the user should know, or do next."""
        if state is None or state.reference_id is None:
            return "Choose a reference tag."

        parts = []
        if state.reference_id in state.visible_ids:
            parts.append(
                f"{DISPLAY_FILTER_SECONDS:g} s median, relative to tag {state.reference_id}."
            )
        else:
            parts.append(f"Reference tag {state.reference_id} is not visible.")

        missing = [
            name
            for name, entry in (("Bush ID", self.bush_entry), ("Cane ID", self.cane_entry))
            if not clean_id(entry.get())
        ]
        if missing:
            parts.append(f"Enter {' and '.join(missing)} to record.")
        return " ".join(parts)

    def _row_values(self, tag_id, state):
        """Return one table row's text."""
        name = self.worker.names.get(tag_id, f"tag_{tag_id}")
        if tag_id in state.relative_positions:
            x, y, z = state.relative_positions[tag_id]
            return (tag_id, name, f"{x:+.3f}", f"{y:+.3f}", f"{z:+.3f}")
        if tag_id not in state.visible_ids:
            return (tag_id, name, "not visible", "", "")
        return (tag_id, name, "--", "--", "--")

    def _update_controls(self, state):
        """Enable controls to match the current state."""
        recording = self.worker.is_recording()
        reference_visible = (
            state is not None
            and state.reference_id is not None
            and state.reference_id in state.visible_ids
        )
        ids_entered = clean_id(self.bush_entry.get()) and clean_id(self.cane_entry.get())
        can_record = reference_visible and ids_entered and not recording and not self.camera_failed
        self.record_button.configure(state="normal" if can_record else "disabled")
        self.reference_box.configure(state="disabled" if recording else "readonly")
        self.hint.configure(text=self._hint_text(state))

        exposure_locked = recording or self.camera_failed or not self.exposure_ready
        self.auto_check.configure(state="disabled" if exposure_locked else "normal")
        scale_locked = exposure_locked or self.auto_exposure.get()
        self.exposure_scale.configure(state="disabled" if scale_locked else "normal")
        if self.auto_exposure.get():
            self.exposure_label.configure(text="auto")
        elif self.exposure_ready and self.exposure_label.cget("text") == "auto":
            self.exposure_label.configure(text=str(round(state.exposure)))

        remaining = state.recording_remaining if state is not None else None
        text = f"Recording {remaining:.1f} s" if recording and remaining is not None else "Record"
        self.record_button.configure(text=text)

    def close(self):
        """Stop the camera and close the window."""
        self.worker.stop()
        self.root.destroy()


def main():
    """Load the config and run the GUI."""
    parser = argparse.ArgumentParser(description="Measure tag poses relative to a reference tag.")
    parser.add_argument("--config", required=True, help="YAML file with tag size and camera settings")
    parser.add_argument("--output-dir", default="data", help="Folder for recordings")
    args = parser.parse_args()

    worker = CaptureWorker(load_config(args.config), output_dir=args.output_dir)
    root = tk.Tk()
    TrackerApp(root, worker)
    worker.start()
    root.mainloop()


if __name__ == "__main__":
    main()
