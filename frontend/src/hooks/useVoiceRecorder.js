/**
 * Voice recorder hook using MediaRecorder API.
 *
 * Supports both tap-to-toggle and press-and-hold interaction patterns.
 * Sends audio/webm chunks over WebSocket as binary frames.
 */
import { useCallback, useRef, useState } from 'react';
import { AUDIO_MIME_TYPE, AUDIO_TIMESLICE_MS } from '../utils/constants';

/**
 * @param {{ sendBinary: Function, sendAudioCommit: Function }} options
 * @returns {{
 *   recording: boolean,
 *   startRecording: () => Promise<void>,
 *   stopRecording: () => void,
 *   toggleRecording: () => Promise<void>,
 *   hasPermission: boolean|null,
 *   error: string|null,
 * }}
 */
export function useVoiceRecorder({ sendBinary, sendAudioCommit }) {
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  const [recording, setRecording] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [error, setError] = useState(null);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 48000,
          channelCount: 1,
        },
      });
      streamRef.current = stream;
      setHasPermission(true);

      // Check for supported MIME type
      const mimeType = MediaRecorder.isTypeSupported(AUDIO_MIME_TYPE)
        ? AUDIO_MIME_TYPE
        : 'audio/webm';

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          sendBinary(event.data);
        }
      };

      recorder.onstop = () => {
        // Send audio commit when recording stops
        sendAudioCommit();
      };

      recorder.start(AUDIO_TIMESLICE_MS);
      setRecording(true);
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setHasPermission(false);
        setError('Microphone permission denied');
      } else {
        setError(`Could not start recording: ${err.message}`);
      }
    }
  }, [sendBinary, sendAudioCommit]);

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }

    // Stop all tracks to release the microphone
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    mediaRecorderRef.current = null;
    setRecording(false);
  }, []);

  const toggleRecording = useCallback(async () => {
    if (recording) {
      stopRecording();
    } else {
      await startRecording();
    }
  }, [recording, startRecording, stopRecording]);

  return {
    recording,
    startRecording,
    stopRecording,
    toggleRecording,
    hasPermission,
    error,
  };
}
