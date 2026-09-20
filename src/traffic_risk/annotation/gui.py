"""Compact temporal annotation GUI for the four portfolio demo videos."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_risk.paths import CONFIG_DIR, PROJECT_ROOT

from .store import AnnotationDocument, Segment


RISK_NAMES = {0: "無風險", 1: "輕微", 2: "中風險", 3: "高風險"}
RISK_COLORS = {0: "#2f9e62", 1: "#e9a23b", 2: "#ef7d32", 3: "#df3b3b"}


def load_case(case_id: str, config_path: Path = CONFIG_DIR / "demo_cases.json") -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {str(value) for value in payload.get("allowed_video_ids", [])}
    if str(case_id) not in allowed:
        raise ValueError(f"case must be one of {sorted(allowed)}")
    return dict(payload["cases"][str(case_id)])


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class TemporalAnnotationApp:
    def __init__(
        self,
        *,
        video_id: str,
        title: str,
        video_path: Path,
        tracks_path: Path | None,
        annotation_path: Path,
    ) -> None:
        try:
            import cv2
            import tkinter as tk
            from tkinter import messagebox, ttk
        except ImportError as exc:  # pragma: no cover - depends on desktop environment
            raise RuntimeError("annotation GUI requires OpenCV and Tkinter") from exc

        if not video_path.is_file():
            raise FileNotFoundError(
                f"video not found: {video_path}\n"
                "Place the four demo videos under data/demo_videos or pass --video."
            )
        self.cv2 = cv2
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.video_id = str(video_id)
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = max(int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.current_frame = 0
        self.playing = False
        self.mark_start: int | None = None
        self.photo = None
        self._slider_guard = False
        self.tracks = self._load_tracks(tracks_path)
        self.track_ids = sorted(self.tracks["track_id"].astype(str).unique().tolist()) if not self.tracks.empty else []
        self.document = AnnotationDocument.load(annotation_path, self.video_id, self.fps)

        self.root = tk.Tk()
        self.root.title(f"Traffic Risk Annotation · Case {self.video_id}")
        self.root.geometry("1420x900")
        self.root.configure(bg="#101722")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.scope_var = tk.StringVar(value="scene")
        self.track_var = tk.StringVar(value=self.track_ids[0] if self.track_ids else "")
        self.status_var = tk.StringVar()
        self._build_ui(title)
        self._bind_keys()
        self.show_frame(0)
        self.refresh_segments()

    def _load_tracks(self, path: Path | None) -> pd.DataFrame:
        required = ["frame", "track_id", "x1", "y1", "x2", "y2"]
        if path is None or not path.is_file():
            return pd.DataFrame(columns=required)
        frame = pd.read_csv(path, low_memory=False)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"tracking CSV is missing columns: {missing}")
        frame = frame.copy()
        frame["frame"] = pd.to_numeric(frame["frame"], errors="coerce").fillna(-1).astype(int)
        frame["track_id"] = frame["track_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        return frame

    def _build_ui(self, title: str) -> None:
        tk, ttk = self.tk, self.ttk
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#101722")
        style.configure("TLabel", background="#101722", foreground="#dce6f3", font=("Noto Sans CJK TC", 11))
        style.configure("Title.TLabel", font=("Noto Sans CJK TC", 18, "bold"), foreground="#ffffff")
        style.configure("TButton", font=("Noto Sans CJK TC", 10), padding=7)
        style.configure("TCombobox", padding=5)

        header = ttk.Frame(self.root, padding=(18, 12))
        header.pack(fill="x")
        ttk.Label(header, text=f"Case {self.video_id}  {title}", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="儲存  Ctrl+S", command=self.save).pack(side="right")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(body, width=300, padding=(14, 0, 0, 0))
        panel.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left, bg="#070b11", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.show_frame(self.current_frame))

        timeline = ttk.Frame(left, padding=(0, 10, 0, 0))
        timeline.pack(fill="x")
        self.play_button = ttk.Button(timeline, text="播放", width=8, command=self.toggle_play)
        self.play_button.pack(side="left")
        self.slider = tk.Scale(
            timeline,
            from_=0,
            to=self.frame_count - 1,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            bg="#101722",
            fg="#ffffff",
            troughcolor="#33445b",
            highlightthickness=0,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Label(timeline, textvariable=self.status_var, width=28).pack(side="right")

        ttk.Label(panel, text="標註層級", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        scope = ttk.Frame(panel)
        scope.pack(fill="x")
        ttk.Radiobutton(
            scope,
            text="整體場景",
            variable=self.scope_var,
            value="scene",
            command=self.refresh_segments,
        ).pack(side="left")
        ttk.Radiobutton(
            scope,
            text="指定車輛",
            variable=self.scope_var,
            value="track",
            command=self.refresh_segments,
        ).pack(side="left", padx=(12, 0))
        self.track_box = ttk.Combobox(panel, textvariable=self.track_var, values=self.track_ids, state="readonly")
        self.track_box.pack(fill="x", pady=(8, 14))
        self.track_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_segments())

        ttk.Button(panel, text="設定區間起點  [", command=self.set_start).pack(fill="x", pady=(0, 8))
        for risk in range(4):
            button = tk.Button(
                panel,
                text=f"{risk}  {RISK_NAMES[risk]}",
                command=lambda value=risk: self.add_segment(value),
                bg=RISK_COLORS[risk],
                fg="white",
                activebackground=RISK_COLORS[risk],
                activeforeground="white",
                relief="flat",
                font=("Noto Sans CJK TC", 11, "bold"),
                pady=8,
            )
            button.pack(fill="x", pady=3)

        ttk.Label(panel, text="已標註區間", style="Title.TLabel").pack(anchor="w", pady=(18, 8))
        self.segment_list = tk.Listbox(
            panel,
            width=38,
            height=16,
            bg="#182231",
            fg="#e9f1fb",
            selectbackground="#3f6fa8",
            relief="flat",
            font=("DejaVu Sans Mono", 10),
        )
        self.segment_list.pack(fill="both", expand=True)
        self.segment_list.bind("<Double-Button-1>", self.seek_selected_segment)
        ttk.Button(panel, text="刪除選取區間", command=self.delete_segment).pack(fill="x", pady=(8, 0))

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.show_frame(self.current_frame - 1))
        self.root.bind("<Right>", lambda _event: self.show_frame(self.current_frame + 1))
        self.root.bind("<bracketleft>", lambda _event: self.set_start())
        for risk in range(4):
            self.root.bind(str(risk), lambda _event, value=risk: self.add_segment(value))
        self.root.bind("<Control-s>", lambda _event: self.save())

    def current_track(self) -> str | None:
        if self.scope_var.get() != "track":
            return None
        track_id = self.track_var.get().strip()
        if not track_id:
            self.messagebox.showwarning("尚未選擇車輛", "請先選擇車輛 ID。")
            return ""
        return track_id

    def _on_slider(self, value: str) -> None:
        if not self._slider_guard:
            self.show_frame(int(float(value)))

    def show_frame(self, frame_index: int) -> None:
        frame_index = max(0, min(int(frame_index), self.frame_count - 1))
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return
        self.current_frame = frame_index
        frame_rows = self.tracks[self.tracks["frame"].eq(frame_index)] if not self.tracks.empty else self.tracks
        selected = self.track_var.get()
        for row in frame_rows.itertuples(index=False):
            track_id = str(row.track_id)
            active = self.scope_var.get() == "track" and track_id == selected
            color = (0, 220, 255) if active else (70, 220, 120)
            thickness = 4 if active else 2
            x1, y1, x2, y2 = (
                int(float(getattr(row, key))) for key in ("x1", "y1", "x2", "y2")
            )
            self.cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            self.cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(18, y1 - 6)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        width = max(self.canvas.winfo_width(), 320)
        height = max(self.canvas.winfo_height(), 240)
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = self.cv2.resize(frame, (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))))
        ok, encoded = self.cv2.imencode(".png", resized)
        if not ok:
            return
        self.photo = self.tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"), format="png")
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.photo, anchor="center")
        self._slider_guard = True
        self.slider.set(frame_index)
        self._slider_guard = False
        time_sec = frame_index / self.fps
        start_text = "未設定" if self.mark_start is None else str(self.mark_start)
        self.status_var.set(f"Frame {frame_index}/{self.frame_count - 1}  {time_sec:.2f}s  起點 {start_text}")

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="暫停" if self.playing else "播放")
        if self.playing:
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing:
            return
        if self.current_frame >= self.frame_count - 1:
            self.playing = False
            self.play_button.configure(text="播放")
            return
        self.show_frame(self.current_frame + 1)
        self.root.after(max(15, int(1000 / min(self.fps, 30.0))), self._play_tick)

    def set_start(self) -> None:
        self.mark_start = self.current_frame
        self.show_frame(self.current_frame)

    def add_segment(self, risk: int) -> None:
        track_id = self.current_track()
        if track_id == "":
            return
        if self.mark_start is None:
            self.mark_start = self.current_frame
        start, end = sorted((self.mark_start, self.current_frame))
        self.document.add(Segment(start, end, risk, self.fps), track_id)
        self.mark_start = end + 1 if end + 1 < self.frame_count else None
        self.refresh_segments()
        self.show_frame(self.current_frame)

    def refresh_segments(self) -> None:
        if not hasattr(self, "segment_list"):
            return
        track_id = self.current_track()
        if track_id == "":
            return
        self.segment_list.delete(0, self.tk.END)
        for segment in self.document.segments(track_id):
            self.segment_list.insert(
                self.tk.END,
                f"{segment.start_frame:6d}-{segment.end_frame:6d}  R{segment.risk} {RISK_NAMES[segment.risk]}",
            )
        self.show_frame(self.current_frame)

    def delete_segment(self) -> None:
        selection = self.segment_list.curselection()
        if not selection:
            return
        track_id = self.current_track()
        if track_id == "":
            return
        self.document.remove(int(selection[0]), track_id)
        self.refresh_segments()

    def seek_selected_segment(self, _event=None) -> None:
        selection = self.segment_list.curselection()
        if not selection:
            return
        track_id = self.current_track()
        if track_id == "":
            return
        self.show_frame(self.document.segments(track_id)[int(selection[0])].start_frame)

    def save(self) -> None:
        self.document.save(self.annotation_path)
        self.messagebox.showinfo("儲存完成", f"已儲存至\n{self.annotation_path}")

    def close(self) -> None:
        self.capture.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_annotation_gui(
    case_id: str,
    *,
    video: Path | None = None,
    tracks: Path | None = None,
    annotation: Path | None = None,
) -> None:
    case = load_case(case_id)
    video_path = video or resolve_project_path(case["video"])
    tracks_path = tracks or resolve_project_path(case["tracks"])
    annotation_path = annotation or resolve_project_path(case["annotation"])
    TemporalAnnotationApp(
        video_id=str(case_id),
        title=str(case["title"]),
        video_path=video_path,
        tracks_path=tracks_path,
        annotation_path=annotation_path,
    ).run()


__all__ = ["TemporalAnnotationApp", "launch_annotation_gui", "load_case"]
