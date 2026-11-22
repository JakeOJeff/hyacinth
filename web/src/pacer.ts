// Frame pacing for a page that both simulates and draws on the GPU.
//
// Two things make WebGPU pages feel laggy: queueing frames faster than the
// GPU finishes them (input latency grows, frames arrive in bursts) and doing
// a fixed amount of work per frame regardless of the machine. The pacer
//   - allows at most `maxInFlight` frames queued on the GPU (default 2, so the
//     CPU prepares frame N+1 while the GPU draws frame N; never more),
//   - measures how long the GPU is actually busy per frame, and
//   - adapts how many simulation steps fit in a frame (additive increase,
//     multiplicative decrease) so frames stay inside a time budget.

export class FramePacer {
  /** Smoothed GPU busy time per frame, ms. */
  gpuMs = 8;
  /** Smoothed interval between animation frames, ms (display refresh). */
  frameInterval = 1000 / 60;
  private inFlight = 0;
  private lastDone = 0;
  private lastFrameAt = 0;
  private cap = 4;

  constructor(
    private readonly device: GPUDevice,
    private readonly maxInFlight = 2,
  ) {}

  /** Call at the start of every animation frame. False = GPU is behind; skip this frame's work. */
  beginFrame(now: number): boolean {
    if (this.lastFrameAt) {
      const dt = Math.min(100, now - this.lastFrameAt);
      this.frameInterval = this.frameInterval * 0.95 + dt * 0.05;
    }
    this.lastFrameAt = now;
    return this.inFlight < this.maxInFlight;
  }

  /** Simulation steps that fit in one frame right now. */
  stepBudget(): number {
    return Math.max(1, Math.floor(this.cap));
  }

  /**
   * Call right after the frame's last queue.submit(). `share` is the fraction
   * of a display frame the GPU may spend: lower keeps headroom for the browser
   * and compositor; higher trades frame rate for throughput (fast-forwarding).
   */
  endFrame(steps: number, share = 0.7): void {
    this.inFlight++;
    const submitted = performance.now();
    this.device.queue.onSubmittedWorkDone().then(() => {
      const now = performance.now();
      // Busy time = from when this frame could start on the GPU until it finished.
      const busy = now - Math.max(submitted, this.lastDone);
      this.lastDone = now;
      this.inFlight--;
      this.gpuMs = this.gpuMs * 0.85 + busy * 0.15;
      if (steps <= 0) return;
      const budget = this.frameInterval * share;
      if (busy > budget * 1.1) this.cap = Math.max(1, this.cap * 0.8);
      else if (busy < budget * 0.8 && steps >= Math.floor(this.cap)) this.cap = Math.min(4096, this.cap * 1.06 + 1);
    });
  }
}
