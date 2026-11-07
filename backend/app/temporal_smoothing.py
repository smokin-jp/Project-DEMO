
from collections import deque


class TemporalKickSmoother:
    def __init__(self, window_size=5, required_kicks=3):
        self.window_size = window_size
        self.required_kicks = required_kicks

        # Each camera session has its own prediction history.
        self.sessions = {}

    def update(self, session_id, kick_detected):
        # Create history for a new camera session.
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(
                maxlen=self.window_size
            )

        history = self.sessions[session_id]

        # Store this frame's prediction.
        history.append(bool(kick_detected))

        # Count positive predictions in the rolling window.
        kick_count = sum(history)

        # Require at least 3 positive predictions.
        smoothed_prediction = (
            kick_count >= self.required_kicks
        )

        return {
            "kick_detected": smoothed_prediction,
            "positive_frames": kick_count,
            "frames_considered": len(history),
            "decision": (
                "FOUL" if smoothed_prediction else "NO FOUL"
            )
        }

    def reset(self, session_id):
        # Clear history when a camera session ends.
        self.sessions.pop(session_id, None)
