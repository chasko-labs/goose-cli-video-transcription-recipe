require('./tracing/otel-config');
const { instrumentVideoProcessing } = require('./tracing/video-tracing');

class OptimizedVideoProcessor {
  async processVideo(videoUrl, videoId) {
    const pipelineSpan = instrumentVideoProcessing(videoId, 'optimized-pipeline');
    try {
      const outputDir = `~/Downloads/video-transcripts/video-${videoId}`;
      await this.download(videoUrl, videoId, outputDir);
      const [transcriptResult, framesResult] = await Promise.all([
        this.transcribe(videoId, outputDir),
        this.extractFrames(videoId, outputDir)
      ]);
      await this.batchAnalyzeFrames(videoId, outputDir);
      await this.generateNarrative(videoId, outputDir);
      pipelineSpan.setStatus({ code: 1 });
    } catch (error) {
      pipelineSpan.setStatus({ code: 2, message: error.message });
      throw error;
    } finally {
      pipelineSpan.end();
    }
  }
}
module.exports = { OptimizedVideoProcessor };
