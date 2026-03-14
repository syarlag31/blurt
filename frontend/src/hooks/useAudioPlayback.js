/**
 * Hook for decoding and playing back audio frames received via WebSocket.
 *
 * Manages an AudioContext, queues incoming audio chunks, and provides
 * visual playback state for the UI.
 *
 * Supports:
 *   - Decoding base64-encoded audio from response.audio messages
 *   - Decoding raw binary audio frames (ArrayBuffer/Blob)
 *   - Queued sequential playback of multiple chunks
 *   - Playback state tracking (playing, idle)
 *   - Stop/cancel playback
 */
import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * @returns {{
 *   playing: boolean,
 *   enqueueAudio: (data: string | ArrayBuffer) => void,
 *   stopPlayback: () => void,
 * }}
 */
export function useAudioPlayback() {
  const [playing, setPlaying] = useState(false);
  const audioCtxRef = useRef(null);
  const queueRef = useRef([]);
  const activeSourceRef = useRef(null);
  const processingRef = useRef(false);
  const stoppedRef = useRef(false);

  // Lazily initialize AudioContext on first use (respects autoplay policies)
  const getAudioContext = useCallback(() => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return null;
      audioCtxRef.current = new AudioCtx();
    }
    // Resume if suspended (browser autoplay policy)
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  // Decode base64 string to ArrayBuffer
  const base64ToArrayBuffer = useCallback((base64) => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }, []);

  // Fallback: play audio via HTMLAudioElement (handles more codecs)
  const playWithAudioElement = useCallback((arrayBuffer) => {
    return new Promise((resolve, reject) => {
      const blob = new Blob([arrayBuffer], { type: 'audio/webm;codecs=opus' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);

      audio.onended = () => {
        URL.revokeObjectURL(url);
        resolve();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Audio playback failed'));
      };

      audio.play().catch((err) => {
        URL.revokeObjectURL(url);
        reject(err);
      });
    });
  }, []);

  // Process the next item in the audio queue
  const processQueue = useCallback(async () => {
    if (processingRef.current) return;
    if (queueRef.current.length === 0) {
      setPlaying(false);
      return;
    }

    processingRef.current = true;
    stoppedRef.current = false;
    setPlaying(true);

    while (queueRef.current.length > 0 && !stoppedRef.current) {
      const audioData = queueRef.current.shift();
      const ctx = getAudioContext();
      if (!ctx) break;

      try {
        // audioData is an ArrayBuffer — decode it
        const audioBuffer = await ctx.decodeAudioData(audioData.slice(0));

        if (stoppedRef.current) break;

        // Play the decoded buffer
        await new Promise((resolve) => {
          const source = ctx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(ctx.destination);
          activeSourceRef.current = source;

          source.onended = () => {
            activeSourceRef.current = null;
            resolve();
          };

          source.start(0);
        });
      } catch {
        // Skip undecodable chunks — could be partial or unsupported format
        // Try playing as a Blob via HTMLAudioElement as fallback
        try {
          await playWithAudioElement(audioData);
        } catch {
          // Truly unplayable — skip silently
        }
      }
    }

    processingRef.current = false;
    if (queueRef.current.length === 0) {
      setPlaying(false);
    }
  }, [getAudioContext, playWithAudioElement]);

  // Enqueue audio data for playback
  const enqueueAudio = useCallback(
    (data) => {
      let arrayBuffer;

      if (typeof data === 'string') {
        // Base64-encoded audio
        arrayBuffer = base64ToArrayBuffer(data);
      } else if (data instanceof ArrayBuffer) {
        arrayBuffer = data;
      } else if (data instanceof Uint8Array) {
        arrayBuffer = data.buffer.slice(
          data.byteOffset,
          data.byteOffset + data.byteLength,
        );
      } else {
        return; // Unknown format
      }

      queueRef.current.push(arrayBuffer);

      // Kick off processing if not already running
      if (!processingRef.current) {
        processQueue();
      }
    },
    [base64ToArrayBuffer, processQueue],
  );

  // Stop all current and queued playback
  const stopPlayback = useCallback(() => {
    stoppedRef.current = true;
    queueRef.current = [];

    if (activeSourceRef.current) {
      try {
        activeSourceRef.current.stop();
      } catch {
        // Already stopped
      }
      activeSourceRef.current = null;
    }

    setPlaying(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stoppedRef.current = true;
      queueRef.current = [];
      if (activeSourceRef.current) {
        try {
          activeSourceRef.current.stop();
        } catch {
          // noop
        }
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
        audioCtxRef.current = null;
      }
    };
  }, []);

  return { playing, enqueueAudio, stopPlayback };
}
