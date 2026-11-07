import { Pipeline } from './pipeline.js';
import { ClipBuffer, compactPoses } from './clip.js';
import { signature } from './appearance.js';

// One camera = its video, its own pose tracker (MediaPipe keeps per-stream tracking state, so each camera needs its own
// instance), detection pipeline and rolling frame buffer. All live cameras stamp frames with performance.now(), so their
// clips line up on one clock.

export class Camera {
  constructor(index, cfg) {
    this.index = index;
    this.id = `cam${index + 1}`;
    this.pipeline = new Pipeline(cfg);
    this.clipBuf = new ClipBuffer();
    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
    this.pose = null;
    this.kind = 'camera'; // 'camera' | 'video'
    this.aspect = 16 / 9;
    this.label = '';
    this.deviceId = '';
    this.lastRes = null;
    this.lastMedia = 0;
    this.looped = false;
  }

  /** which side RED is on in this camera's view (a camera on the opposite side sees the fighters the other way round) */
  setFlip(flip) {
    this.pipeline.tracker.flip = !!flip;
    this.pipeline.reset();
  }

  async open(deviceId = '') {
    const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    const attempts = [
      deviceId ? { video: { ...base, deviceId: { exact: deviceId } } } : { video: base },
      deviceId ? { video: { deviceId: { exact: deviceId } } } : { video: true },
    ];
    let stream, lastErr;
    for (const c of attempts) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ ...c, audio: false });
        break;
      } catch (e) {
        lastErr = e;
        if (e.name !== 'OverconstrainedError' && e.name !== 'NotReadableError') break; // permission / no device: retrying won't help
      }
    }
    if (!stream) throw lastErr;
    this.video.srcObject = stream;
    await this.video.play();
    const track = stream.getVideoTracks()[0];
    this.deviceId = deviceId || track?.getSettings().deviceId || ''; // keep the id the user picked so the dropdown stays on it
    this.label = track?.label || this.id;
    this.aspect = this.video.videoWidth / this.video.videoHeight || 16 / 9;
  }

  async openFile(file, rate = 1) {
    this.kind = 'video';
    this.video.src = URL.createObjectURL(file);
    this.video.loop = true;
    this.video.playbackRate = rate;
    await this.video.play();
    this.label = file.name;
    this.aspect = this.video.videoWidth / this.video.videoHeight || 16 / 9;
  }

  stop() {
    this.video.srcObject?.getTracks().forEach((t) => t.stop());
    this.video.pause();
    if (this.video.src.startsWith('blob:')) URL.revokeObjectURL(this.video.src);
    this.video.removeAttribute('src');
    this.video.srcObject = null;
    this.pose?.close?.();
    this.pose = null;
    this.clipBuf.reset();
  }

  /** torso colour signature of each detected person (a small downscaled copy of the frame is enough) */
  sampleColours(poses) {
    const W = 320, H = Math.max(8, Math.round(W / this.aspect));
    if (!this.sc) {
      this.sc = document.createElement('canvas');
      this.sg = this.sc.getContext('2d', { willReadFrequently: true });
    }
    if (this.sc.width !== W || this.sc.height !== H) {
      this.sc.width = W;
      this.sc.height = H;
    }
    this.sg.drawImage(this.video, 0, 0, W, H);
    const data = this.sg.getImageData(0, 0, W, H).data;
    return poses.slice(0, 2).map((p) => signature(data, W, H, p.lm));
  }

  /** one detection step. Returns the pipeline result, or null when there is no new frame */
  step() {
    const v = this.video;
    if (!this.pose || v.readyState < 2) return null;
    const media = this.kind === 'video' ? v.currentTime * 1000 : performance.now();
    this.looped = false;
    if (this.kind === 'video') {
      if (media < this.lastMedia - 300) this.looped = true; // clip restarted
      if (media === this.lastMedia) return null;
      this.lastMedia = media;
    }
    const poses = this.pose.detect(v);
    const sigs = this.pipeline.tracker.appearance && poses.length ? this.sampleColours(poses) : undefined;
    const res = this.pipeline.process({ t: media, aspect: this.aspect, poses, sigs });
    this.clipBuf.push(media, v, compactPoses(res.people, this.aspect));
    this.lastRes = res;
    return res;
  }
}
