"""Measure tag poses relative to a chosen reference tag.

Run from the repository root:
    python -m src.gui --config config/example.yaml
"""

import argparse
import math
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.capture import CaptureWorker
from src.config import load_config
from src.measurement import (
    DISPLAY_FILTER_SECONDS,
    STATUS_FAIR,
    STATUS_GOOD,
    STATUS_NONE,
    QualityLimits,
    clean_id,
    judge_quality,
    tag_name,
)

# ---- Detection quality: row colours in the tag table -------------------------
# Each tag is judged over the last QUALITY_WINDOW_SECONDS, in the camera frame.
# A recording is judged by the same limits over its whole length: a tag is saved as
# well_detected in the CSV, and checked off, if it and the reference were green.
#   Green:     detected in at least GREEN_MIN_DETECTED of frames, at most
#              GREEN_MAX_FLIPPED of those flipped, and jitter at most GREEN_MAX_JITTER_MM.
#   Yellow:    detected, but missing at least one green limit.
#   No colour: detected in under MIN_DETECTED of frames, or not seen for NOT_SEEN_SECONDS.
QUALITY_WINDOW_SECONDS = 2.0
MIN_DETECTED = 0.20             # share of frames
GREEN_MIN_DETECTED = 0.70       # share of frames
# A flip is a frame more than ROTATION_OUTLIER_DEG (15 degrees) from the tag's typical
# rotation, the same test the CSV uses to drop frames.
GREEN_MAX_FLIPPED = 0.15        # share of detections
# Robust standard deviation of position on the worst axis, after removing flips.
# It averages out over a recording: the CSV's error is roughly jitter / sqrt(frames).
GREEN_MAX_JITTER_MM = 10.0
NOT_SEEN_SECONDS = 1.0
# A row's colour only changes once the new colour has lasted this long, to stop flicker.
STATUS_HOLD_SECONDS = 1.0
# Only allow recording while the reference tag's row is green.
REQUIRE_GOOD_REFERENCE = True
COLOR_GOOD = "#c8e6c9"
COLOR_FAIR = "#fff1a8"
# Tags already recorded green for this cane are greyed out and checked off.
COLOR_DONE_BACKGROUND = "#ececec"
COLOR_DONE_TEXT = "#9a9a9a"
CHECKMARK = "\u2714"
COLOR_ALL_RECORDED = "#1b7a2b"  # text saying every tag is recorded
# -----------------------------------------------------------------------------

QUALITY_LIMITS = QualityLimits(
    min_detected=MIN_DETECTED,
    green_min_detected=GREEN_MIN_DETECTED,
    green_max_flipped=GREEN_MAX_FLIPPED,
    green_max_jitter_mm=GREEN_MAX_JITTER_MM,
    not_seen_seconds=NOT_SEEN_SECONDS,
)
STATUS_DONE = "done"

POLL_MILLISECONDS = 30
# Wait this long after the slider stops moving before changing the camera's exposure.
# Sending a change for every step of a drag can make the camera stop responding.
EXPOSURE_SEND_DELAY_MILLISECONDS = 300
# How long closing the window waits for the camera to be released. A camera that
# has stopped responding can block Intel's library for minutes, so after this the
# program ends itself rather than staying open in the background.
CAMERA_RELEASE_SECONDS = 10.0
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


class StatusHold:
    """Show a new status only once it has lasted a while, so colours don't flicker."""

    def __init__(self, hold_seconds=STATUS_HOLD_SECONDS):
        """Start with nothing shown."""
        self.hold_seconds = hold_seconds
        self.shown = {}    # key -> status on screen
        self.pending = {}  # key -> (status, time it first appeared)

    def update(self, key, status, now):
        """Report the latest status for a row and return the one to show."""
        if key not in self.shown or status == self.shown[key]:
            self.shown[key] = status
            self.pending.pop(key, None)
            return status
        pending_status, since = self.pending.get(key, (None, now))
        if pending_status != status:
            self.pending[key] = (status, now)
        elif now - since >= self.hold_seconds:
            self.shown[key] = status
            del self.pending[key]
        return self.shown[key]


class TrackerApp:
    """Window showing the camera, live tag positions, and recording controls."""

    def __init__(self, root, worker):
        """Build the window and start polling the worker."""
        self.root = root
        self.worker = worker
        self.last_sequence = 0
        self.exposure_job = None  # Pending slider change, sent once the slider settles.
        self.camera_released = True
        self.was_connected = False
        self.photo = None  # Tk only shows an image while a reference to it exists.
        self.rows = {}     # tag_id -> table row
        self.reference_row = None  # table row for the reference tag, always first
        self.reference_judgement = (STATUS_NONE, "")  # (held status, detection text)
        self.row_contents = {}  # table row -> (values, tags) last shown, to skip redraws
        self.status_hold = StatusHold()
        self.recorded = set()  # tags recorded green for this cane
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

        columns = ("tag", "name", "x", "y", "z", "detection", "recorded")
        self.table = ttk.Treeview(panel, columns=columns, show="headings", height=14)
        headings = {
            "tag": "Tag", "name": "Name", "x": "x (m)", "y": "y (m)", "z": "z (m)",
            "detection": "Detection", "recorded": "Recorded?",
        }
        widths = {
            "tag": 45, "name": 90, "x": 80, "y": 70, "z": 70, "detection": 155, "recorded": 90,
        }
        anchors = {"x": "e", "y": "e", "z": "e", "recorded": "center"}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor=anchors.get(column, "w"))
        self.table.tag_configure(STATUS_GOOD, background=COLOR_GOOD)
        self.table.tag_configure(STATUS_FAIR, background=COLOR_FAIR)
        self.table.tag_configure(
            STATUS_DONE, background=COLOR_DONE_BACKGROUND, foreground=COLOR_DONE_TEXT
        )
        self.table.grid(row=1, column=0, sticky="ns")

        hint_area = ttk.Frame(panel)
        hint_area.grid(row=2, column=0, sticky="w", pady=(6, 10))
        self.hint = ttk.Label(hint_area, text="Choose a reference tag.", wraplength=470)
        self.hint.pack(anchor="w")
        # Shown in green, below the hint, once every tag seen so far is checked off.
        self.all_recorded_label = ttk.Label(
            hint_area,
            text="Every tag seen so far is recorded. Press New cane to start the next one.",
            foreground=COLOR_ALL_RECORDED,
            wraplength=470,
        )

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
        self.new_cane_button = ttk.Button(id_row, text="New cane", command=self._on_new_cane)
        self.new_cane_button.pack(side="right")

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
        """Show the slider's exposure, and send it once the slider stops moving."""
        low, high = self.worker.get_state().exposure_range
        exposure = slider_to_exposure(float(position), low, high)
        self.exposure_label.configure(text=str(exposure))
        self._cancel_exposure_send()
        self.exposure_job = self.root.after(
            EXPOSURE_SEND_DELAY_MILLISECONDS, self._send_exposure, exposure
        )

    def _send_exposure(self, exposure):
        """Send a settled slider value to the worker."""
        self.exposure_job = None
        self.worker.set_exposure(exposure)

    def _cancel_exposure_send(self):
        """Drop a slider change that hasn't been sent yet."""
        if self.exposure_job is not None:
            self.root.after_cancel(self.exposure_job)
            self.exposure_job = None

    def _flush_exposure_send(self):
        """Send a slider change that hasn't been sent yet, straight away."""
        if self.exposure_job is not None:
            self._cancel_exposure_send()
            low, high = self.worker.get_state().exposure_range
            self.worker.set_exposure(slider_to_exposure(float(self.exposure_scale.get()), low, high))

    def _on_auto_exposure(self):
        """Send the Auto checkbox's value to the worker."""
        # A late slider change would otherwise turn auto exposure back off.
        self._cancel_exposure_send()
        self.worker.set_auto_exposure(self.auto_exposure.get())

    def _on_record(self):
        """Validate the recording length and start a recording."""
        # Exposure changes are ignored once recording starts, so send any pending one first.
        self._flush_exposure_send()
        try:
            seconds = float(self.seconds_box.get())
            self.worker.start_recording(seconds, self.bush_entry.get(), self.cane_entry.get())
        except ValueError:
            self._set_status("Enter a recording length above 0 seconds.", is_error=True)
            return
        self.record_button.configure(state="disabled")
        self._set_status("Recording...")

    def _on_new_cane(self):
        """Clear the checkmarks and the Cane ID, after confirming if any tags are checked."""
        if self.recorded and not messagebox.askyesno(
            "New cane",
            f"Clear the checkmarks for {len(self.recorded)} recorded tags, and the Cane ID?",
            parent=self.root,
        ):
            return
        self.recorded.clear()
        self.cane_entry.delete(0, "end")
        self.cane_entry.focus_set()
        self._set_status("Checkmarks cleared for a new cane. Enter its Cane ID.")

    def _check_off(self, verdict):
        """Check off the tags a saved recording captured well; return a message saying which."""
        if verdict.reference_problem:
            return (
                f"No tags checked off: the reference wasn't detected well "
                f"({verdict.reference_problem})."
            )
        self.recorded.update(verdict.well_detected)
        parts = []
        if verdict.well_detected:
            parts.append(f"Checked off: {self._tag_list(verdict.well_detected)}.")
        if verdict.problems:
            reasons = ", ".join(
                f"{self._tag_name(tag_id)} ({text})" for tag_id, text in verdict.problems.items()
            )
            parts.append(f"Not good enough: {reasons}.")
        return " ".join(parts) or "No tags were seen with the reference."

    def _tag_name(self, tag_id):
        """Return a tag's display name."""
        return tag_name(self.worker.names, tag_id)

    def _tag_list(self, tag_ids):
        """Return tag names as a comma-separated list, in ID order."""
        return ", ".join(self._tag_name(tag_id) for tag_id in sorted(tag_ids))

    def _set_status(self, text, is_error=False):
        """Show a message in the status bar."""
        self.status_label.configure(text=text, foreground=COLOR_ERROR if is_error else COLOR_NORMAL)

    def poll(self):
        """Show the worker's latest results, then schedule the next poll."""
        self._show_notices()
        state = self.worker.get_state()
        self._show_connection(state)
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
        """Show worker messages, joining those that arrive together.

        A saved recording's notice carries its verdict, so its checklist message is
        shown with it.
        """
        notices = self.worker.get_notices()
        if not notices:
            return
        texts = []
        for notice in notices:
            texts.append(notice.text)
            if notice.verdict is not None:
                texts.append(self._check_off(notice.verdict))
        has_error = any(notice.is_error for notice in notices)
        self._set_status("  ".join(texts), is_error=has_error)

    def _show_connection(self, state):
        """Hide the last frame and say so while the camera is disconnected."""
        connected = self.worker.is_connected()
        if connected and not self.was_connected:
            # Match the exposure controls to the camera again, as a failed change may
            # have left them showing a value the camera isn't using.
            self.exposure_ready = False
        self.was_connected = connected
        if connected:
            self.canvas.itemconfigure(self.canvas_image, state="normal")
            return
        message = "Starting camera..." if state is None else "Camera disconnected. Reconnecting..."
        self.canvas.itemconfigure(self.canvas_image, state="hidden")
        self.canvas.itemconfigure(self.canvas_text, text=message)

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
        """Show the reference tag first, then every other tag, coloured by detection."""
        now = time.monotonic()
        tag_ids = [tag_id for tag_id in state.seen_ids if tag_id != state.reference_id]
        for tag_id in set(self.rows) - set(tag_ids):
            row = self.rows.pop(tag_id)
            self.table.delete(row)
            self.row_contents.pop(row, None)

        for tag_id in tag_ids:
            status, text = self._judge(tag_id, state, now)
            check = ""
            if tag_id in self.recorded:
                status, check = STATUS_DONE, CHECKMARK
            values = (*self._row_values(tag_id, state), text, check)
            if tag_id in self.rows:
                self._set_row(self.rows[tag_id], values, (status,))
            else:
                position = sum(other < tag_id for other in self.rows)
                self.rows[tag_id] = self.table.insert("", position, values=values, tags=(status,))

        self._update_reference_row(state, now)

    def _update_reference_row(self, state, now):
        """Keep the reference tag's row at the top, with its own detection status."""
        if state.reference_id is None:
            if self.reference_row is not None:
                self.table.delete(self.reference_row)
                self.row_contents.pop(self.reference_row, None)
                self.reference_row = None
            self.reference_judgement = (STATUS_NONE, "")
            return

        status, text = self._judge(state.reference_id, state, now)
        self.reference_judgement = (status, text)
        values = (state.reference_id, "reference", "origin", "", "", text, "")
        if self.reference_row is None:
            self.reference_row = self.table.insert("", 0, values=values, tags=(status,))
        else:
            self._set_row(self.reference_row, values, (status,))
            if self.table.index(self.reference_row) != 0:
                self.table.move(self.reference_row, "", 0)

    def _set_row(self, row, values, tags):
        """Update a table row, skipping it if nothing changed, since each change redraws the table."""
        if self.row_contents.get(row) != (values, tags):
            self.row_contents[row] = (values, tags)
            self.table.item(row, values=values, tags=tags)

    def _judge(self, tag_id, state, now):
        """Return a tag's status, held steady against flicker, and its detection text."""
        status, text = judge_quality(state.quality.get(tag_id), self.worker.quality_limits)
        return self.status_hold.update(tag_id, status, now), text

    def _set_up_exposure(self, state):
        """Match the exposure controls to the camera on the first frame."""
        if self.exposure_ready:
            return
        self.exposure_ready = True
        low, high = state.exposure_range
        # Set with no command attached, so this doesn't send a change.
        self._cancel_exposure_send()
        self.exposure_scale.configure(state="normal", command="")
        self.exposure_scale.set(exposure_to_slider(state.exposure, low, high))
        self.exposure_scale.configure(command=self._on_exposure_slider)
        self.exposure_label.configure(text=str(round(state.exposure)))
        self.auto_exposure.set(state.auto_exposure)

    def _hint_text(self, state):
        """Return what the user should know, or do next."""
        if state is None or state.reference_id is None:
            return "Choose a reference tag."

        parts = []
        if state.reference_id not in state.visible_ids:
            parts.append(f"Reference tag {state.reference_id} is not visible.")
        else:
            parts.append(
                f"{DISPLAY_FILTER_SECONDS:g} s median, relative to tag {state.reference_id}."
            )
            status, text = self.reference_judgement
            if status != STATUS_GOOD:
                problem = f" ({text})" if status == STATUS_FAIR else ""
                needed = " Recording needs it green." if REQUIRE_GOOD_REFERENCE else ""
                parts.append(f"The reference isn't detected well{problem}.{needed}")

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
        name = self._tag_name(tag_id)
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
        reference_ok = self.reference_judgement[0] == STATUS_GOOD or not REQUIRE_GOOD_REFERENCE
        ids_entered = clean_id(self.bush_entry.get()) and clean_id(self.cane_entry.get())
        connected = self.worker.is_connected()
        can_record = (
            reference_visible and reference_ok and ids_entered and not recording and connected
        )
        self.record_button.configure(state="normal" if can_record else "disabled")
        self.reference_box.configure(state="disabled" if recording else "readonly")
        self.new_cane_button.configure(state="disabled" if recording else "normal")
        self.hint.configure(text=self._hint_text(state))
        all_recorded = bool(self.rows) and set(self.rows) <= self.recorded
        if all_recorded and not self.all_recorded_label.winfo_manager():
            self.all_recorded_label.pack(anchor="w", pady=(4, 0))
        elif not all_recorded and self.all_recorded_label.winfo_manager():
            self.all_recorded_label.pack_forget()

        exposure_locked = recording or not connected or not self.exposure_ready
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
        self._cancel_exposure_send()
        self.camera_released = self.worker.stop(timeout=CAMERA_RELEASE_SECONDS)
        self.root.destroy()


def main():
    """Load the config and run the GUI."""
    parser = argparse.ArgumentParser(description="Measure tag poses relative to a reference tag.")
    parser.add_argument("--config", required=True, help="YAML file with tag size and camera settings")
    parser.add_argument("--output-dir", default="data", help="Folder for recordings")
    args = parser.parse_args()

    worker = CaptureWorker(
        load_config(args.config),
        output_dir=args.output_dir,
        quality_window_seconds=QUALITY_WINDOW_SECONDS,
        quality_limits=QUALITY_LIMITS,
    )
    root = tk.Tk()
    app = TrackerApp(root, worker)
    worker.start()
    root.mainloop()
    exit_if_camera_stuck(app)


def exit_if_camera_stuck(app):
    """End the program at once if closing the window couldn't release the camera.

    A normal exit would wait on the stuck camera thread, possibly for minutes,
    while still holding the camera. Ending the process makes the operating system
    release it. Recordings are always fully saved before the window can close.
    """
    if app.camera_released:
        return
    print(
        "The camera did not respond while closing, so the program was ended to release it. "
        "If it happens again, unplug the camera and plug it back in.",
        file=sys.stderr,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


if __name__ == "__main__":
    main()
