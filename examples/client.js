const BARGEIN_THRESHOLD = 0; // this is how it is set on the environment. 
const BARGEIN_DURATION_MS = 2000; // NEW: Trigger bargein after 2000ms of speech

// Smart barge-in function that considers confidence for all interim results
function shouldTriggerSmartBargeIn(transcript, confidence, isFirstInterim, mediaStream) {
    const wordCount = transcript.split(/\s+/).length;
  
    mediaStream.logger.info('Word count:', { wordCount: wordCount });
    // Must have enough valid words first
    if (wordCount <= BARGEIN_THRESHOLD) {
      mediaStream.logger.info('[Interim] Rejecting barge in because word count is less than threshold. ', { transcript: transcript });
      return false;
    }

    // NEW: Check audio duration threshold
    const speechDurationMs = confidence?.audio_duration_ms || 0;
    if (speechDurationMs < BARGEIN_DURATION_MS) {
      mediaStream.logger.info(`[Interim] Rejecting barge in - duration (${speechDurationMs}ms) < threshold (${BARGEIN_DURATION_MS}ms)`);
      return false;
    }
  
    // Apply confidence filtering to ALL interim results, not just first
    if (confidence && confidence.avg_confidence) {
      const MIN_CONFIDENCE_FOR_FIRST_WORD = 0.70; // For first word after silence, default was 0.8
      const MIN_CONFIDENCE_FOR_ONGOING = 0.70; // For subsequent words (more lenient), default was 0.8
  
      const confidenceThreshold = isFirstInterim ? MIN_CONFIDENCE_FOR_FIRST_WORD : MIN_CONFIDENCE_FOR_ONGOING;
      if (confidence.avg_confidence < confidenceThreshold) {
        mediaStream.logger.info(`[Interim] Rejecting barge in because confidence is less than threshold. Received avg confidence: ${confidence.avg_confidence}, threshold: ${confidenceThreshold}`);
        return false;
      }
    }
  
    return true;
  }

// Viet, 
// mediaStream ->> is an object that is related to the state of the call. It contains messages, etc. Not needed for your testing I think. 
// sttHost ->> is the host of the stt when planned to be used on an external gpu. Also not related to your testing. 
const setupJobixSTT = async (mediaStream, sttHost) => {
    return new Promise((resolve, reject) => {
      const sessionId = uuidv4();
      // X-Word-Confidence-Threshold=0.85 <<- original
      // X-Endpointing-MS=400
      // X-VAD-Aggressiveness=3
      // X-Max-Speech-Segment-MS=15000
      // X-Min-Speech-Duration-MS=250
      const ws = new WebSocket("ws://" + sttHost + ":8089/ws/stt"
        + "?session_id=" + sessionId
        + "&interim_results=true"
        + "&encoding=mulaw_8k"
        + "&X-Endpointing-MS=500"
        + "&X-VAD-Frame-MS=30"
        + "&X-VAD-Aggressiveness=3"
        + "&X-Max-Speech-Segment-MS=25000"
        + "&X-Min-Speech-Duration-MS=500" 
        + "&X-Word-Confidence-Threshold=0.85"
        + "&X-Short-Transcript-Max-Len=3"
        + "&X-Short-Transcript-Avg-Word-Confidence-Threshold=0.70"
        + "&X-Single-Word-Confidence-Threshold=0.85"
      );
  
      // Initialize transcript buffer for interim results
      mediaStream.transcriptBuffer = [];
  
      ws.onopen = () => {
        mediaStream.logger.info("Jobix STT: WebSocket connected");
        mediaStream.jobixSocket = ws;
        resolve(ws);
      };
  
      ws.onerror = (err) => {
        mediaStream.logger.error("Jobix STT: WebSocket error", { message: err.message });
        reject(err);
      };
  
  
  
      ws.onmessage = async (message) => {
        const data = JSON.parse(message.data);
  
        if (data.command === "please_repeat_message") {
          mediaStream.logger.info('Jobix STT: Please repeat message received. Note:(method deprecated)');
          return;
        }
  
        let transcript = data.transcript;
        const isFinal = data.is_final;
  
        mediaStream.logger.info('Jobix STT: Transcript received:', { data: data });
  
        // isFinal is true >> pass the transcript even if it is empty
  
        if (
          (transcript !== "" && !isFinal) ||
          (transcript == "" && isFinal) ||
          (transcript !== "" && isFinal)) {
  
          if (mediaStream.sttStartTime === null) {
            mediaStream.sttStartTime = Date.now();
            mediaStream.logger.info(`Jobix STT process started at: ${mediaStream.sttStartTime}`);
          }
  
          if (!isFinal) {
  
            mediaStream.transcriptBuffer.push(transcript);
  
            if (shouldTriggerSmartBargeIn(transcript, data, true, mediaStream) && !mediaStream.bargeIn) {
              // we will allow barge in
              mediaStream.bargeIn = true;
              mediaStream.logger.info('[INTERIM] Barge in detected (interim):', {
                transcript: transcript,
                totalWords: transcript.split(/\s+/).length,
                confidence: data.avg_confidence || 'N/A',
                bargeIn: mediaStream.bargeIn
              });
            }
  
            mediaStream.logger.info('[INTERIM] AgentMessageFinished and bargeIn:', { agentMessageFinished: mediaStream.agentMessageFinished, bargeIn: mediaStream.bargeIn });
            if (!mediaStream.agentMessageFinished && !mediaStream.bargeIn) {
              // we exit the function
              mediaStream.logger.info('[INTERIM] Agent is talking and bargeIn is false. We ignore the message.');
              mediaStream.logger.info('[INTERIM] Barge in value:', { bargeIn: mediaStream.bargeIn });
              mediaStream.sttStartTime = null;
              return;
            }
            if (!mediaStream.bargeIn) {
              mediaStream.sttStartTime = null;
            }
            mediaStream.logger.info('[INTERIM] Agent message status:', { agentMessageFinished: mediaStream.agentMessageFinished });
            (async () => {
              if (mediaStream.speaking && mediaStream.bargeIn && !mediaStream.agentBusyCallingTool) {
                mediaStream.regularAssistantMessageID = null;
                mediaStream.logger.info('[INTERIM] Clearing audio playback');
                if (mediaStream.ttsProvider === 'jobixtts') {
                  mediaStream.logger.info('[INTERIM] Clearing audio buffer jobixTTS:');
                  mediaStream.logger.info('[INTERIM] Tracking id that will be cleared: ', { ttsTrackingIDs: Array.from(mediaStream.ttsTrackingIDs) });
                  mediaStream.ttsWebsocket.send(JSON.stringify({ 'command': 'clear-buffer' }));
                  mediaStream.ttsTrackingIDs.clear();
                  mediaStream.logger.info('[INTERIM] Cleared tracking ID list due to barge-in', { ttsTrackingIDsCount: mediaStream.ttsTrackingIDs.size });
                  // Abort current LLM generation if ongoing
                  const controller = mediaStream.streamControllers.get(mediaStream.requestId);
                  if (controller) {
                    controller.abort();
                    mediaStream.logger.info('[INTERIM] Aborted Gemini/OpenAI stream due to barge-in', { requestId: mediaStream.requestId });
                    mediaStream.streamControllers.delete(mediaStream.requestId);
                    mediaStream.validRequestIds.delete(mediaStream.requestId);
                    mediaStream.requestId = null;
                  }
                  mediaStream.logger.info('[INTERIM] ttsRtpBuffer size before clearing: ', { ttsRtpBufferSize: mediaStream.ttsRtpBuffer.length });
                  mediaStream.ttsRtpBuffer = []; // Drop remaining audio in buffer
                  mediaStream.logger.info('[INTERIM] ttsRtpBuffer emptied: ', { ttsRtpBuffer: mediaStream.ttsRtpBuffer });
                } else {
                  mediaStream.ttsWebsocket.send(JSON.stringify({ 'text': ' ', 'flush': true }));
                  mediaStream.logger.info('[INTERIM] ttsRtpBuffer size before clearing: ', { ttsRtpBufferSize: mediaStream.ttsRtpBuffer.length });
                  mediaStream.ttsRtpBuffer = []; // Drop remaining audio in buffer
                  mediaStream.logger.info('[INTERIM] ttsRtpBuffer emptied: ', { ttsRtpBuffer: mediaStream.ttsRtpBuffer });
                }
                mediaStream.logger.info('[INTERIM] rtp_handler: clear audio playback', { streamSid: mediaStream.streamSid });
                // Handles Barge In
                const messageJSON = {
                  event: 'clear',
                };
                mediaStream.connection.send(JSON.stringify(messageJSON));
                mediaStream.logger.debug('[INTERIM] Clear payload sent to rtp_handler', { messageJSON: messageJSON });
                // mediaStream.deepgramTTSWebsocket.send(JSON.stringify({ 'type': 'Clear' }));
                mediaStream.speaking = false;
                mediaStream.bargeIn = false;
                mediaStream.agentMessageFinished = true;
                mediaStream.logger.info('[INTERIM] Barge in value:', { bargeIn: mediaStream.bargeIn });
              }
              // Cancel ongoing LLM request ONLY if agent is not calling a tool
              if (!mediaStream.agentBusyCallingTool && mediaStream.requestId) {
                mediaStream.logger.info('[INTERIM] User interrupted assistant. Aborting LLM generation.', { cancelling_RequestId: mediaStream.requestId });
                mediaStream.validRequestIds.delete(mediaStream.requestId);
                mediaStream.requestId = null; // Reset requestId
                mediaStream.regularAssistantMessageID = null;
              }
            })();
            // Reset timers even for interim results (user is actively speaking)
            if (mediaStream.noResponseTimer) {
              clearTimeout(mediaStream.noResponseTimer);
              mediaStream.noResponseTimer = null;
            }
            if (mediaStream.waitUserAnswerTimer) {
              clearTimeout(mediaStream.waitUserAnswerTimer);
              mediaStream.waitUserAnswerTimer = null;
            }
            if (mediaStream.onActionTimer) {
              clearTimeout(mediaStream.onActionTimer);
              mediaStream.onActionTimer = null;
            }
  
            return; // Don't process interim results further
          }
  
          // Final result - use the final transcript (interim results are partial, final is complete)
          if (transcript.trim() === '' && mediaStream.transcriptBuffer.length > 0) {
            transcript = mediaStream.transcriptBuffer.join(' ');
            mediaStream.logger.info('[IS_FINAL] Final transcript is empty, using buffered interim transcript.', { transcript });
          }
  
          if (transcript.trim() === '') {
            mediaStream.logger.info('[IS_FINAL] Final transcript is empty, ignoring.');
            mediaStream.sttStartTime = null;
            return;
          }
  
          if (mediaStream.speaking) {
            if (transcript.split(/\s+/).length > BARGEIN_THRESHOLD) {
              mediaStream.bargeIn = true;
              mediaStream.logger.info('[IS_FINAL] Barge in value:', { bargeIn: mediaStream.bargeIn });
            }
          }
  
          if (mediaStream.transcriptBuffer.length > 0) {
            mediaStream.logger.info('[IS_FINAL] Final result received after interim progression:', {
              finalTranscript: transcript,
              interimCount: mediaStream.transcriptBuffer.length,
              interimProgression: mediaStream.transcriptBuffer
            });
            // Clear the buffer (interim results were just partial previews)
            mediaStream.transcriptBuffer = [];
          } else {
            mediaStream.logger.info('[IS_FINAL] Final result (no interim buffered):', { transcript: transcript });
          }
  
  
          mediaStream.logger.info('[IS_FINAL] Processing final transcript:', { transcript: transcript });
  
          // Reset the no-response timer if the user is speaking
          if (mediaStream.noResponseTimer) {
            clearTimeout(mediaStream.noResponseTimer);
            mediaStream.noResponseTimer = null;
          }
          if (mediaStream.waitUserAnswerTimer) {
            clearTimeout(mediaStream.waitUserAnswerTimer);
            mediaStream.waitUserAnswerTimer = null;
          }
          if (mediaStream.onActionTimer) {
            clearTimeout(mediaStream.onActionTimer);
            mediaStream.onActionTimer = null;
          }
  
          // while call is under 1 minute we will check if user is not a machine
          if (mediaStream.agentCallLimitTime - mediaStream.call_duration_limit_ms < 60500) {
            mediaStream.logger.info('call length is: ', { call_length: mediaStream.agentCallLimitTime - mediaStream.call_duration_limit_ms });
            if (voicemailKeywords.some(phrase => transcript.toLowerCase().includes(phrase.toLowerCase()))) {
              mediaStream.logger.info('Anti VM detected. Ending the call. Found phrase:', { phrase: voicemailKeywords.find(phrase => transcript.toLowerCase().includes(phrase.toLowerCase())) });
              mediaStream.agentGoodbyeMessageID = 'anti_machine_detected';
              mediaStream.logger.info('Closing the call...');
              // Send a close event before closing the connection
              const closeEvent = {
                type: "utf8",
                utf8Data: JSON.stringify({
                  event: "close",
                  reason: "agent_ended_call"
                })
              };
              mediaStream.connection.send(JSON.stringify(closeEvent));
              mediaStream.logger.info('sent close event to rtp_handler, reason: anti_machine_detected');
  
              mediaStream.connection.close();
              mediaStream.close();
              mediaStream.closeThisCall = true;
              return;
            }
          }
  
          mediaStream.logger.info('[IS_FINAL] Agent message status:', { agentMessageFinished: mediaStream.agentMessageFinished });
          if (!mediaStream.agentMessageFinished && !mediaStream.bargeIn) {
            // we exit the function
            mediaStream.logger.info('[IS_FINAL] Agent is talking and bargeIn is false. We ignore the message.');
            mediaStream.logger.info('[IS_FINAL] Barge in value:', { bargeIn: mediaStream.bargeIn });
            return;
          }
  
          (async () => {
            if (mediaStream.speaking && mediaStream.bargeIn && !mediaStream.agentBusyCallingTool) {
              mediaStream.regularAssistantMessageID = null;
              mediaStream.logger.info('[IS_FINAL] Clearing audio playback');
              if (mediaStream.ttsProvider === 'jobixtts') {
                mediaStream.logger.info('[IS_FINAL] Clearing audio buffer jobixTTS:');
                mediaStream.logger.info('[IS_FINAL] Tracking id that will be cleared: ', { ttsTrackingIDs: Array.from(mediaStream.ttsTrackingIDs) });
                mediaStream.ttsWebsocket.send(JSON.stringify({ 'command': 'clear-buffer' }));
                mediaStream.ttsTrackingIDs.clear();
                mediaStream.logger.info('[IS_FINAL] Cleared tracking ID list due to barge-in', { ttsTrackingIDsCount: mediaStream.ttsTrackingIDs.size });
                // Abort current LLM generation if ongoing
                const controller = mediaStream.streamControllers.get(mediaStream.requestId);
                if (controller) {
                  controller.abort();
                  mediaStream.logger.info('[IS_FINAL] Aborted Gemini/OpenAI stream due to barge-in', { requestId: mediaStream.requestId });
                  mediaStream.streamControllers.delete(mediaStream.requestId);
                  mediaStream.validRequestIds.delete(mediaStream.requestId);
                  mediaStream.requestId = null;
                }
              } else {
                mediaStream.ttsWebsocket.send(JSON.stringify({ 'text': ' ', 'flush': true }));
              }
  
              // Handles Barge In
              const messageJSON = {
                event: 'clear',
              };
              mediaStream.connection.send(JSON.stringify(messageJSON));
              mediaStream.logger.info('[IS_FINAL] Clear payload sent to frontend', { messageJSON: messageJSON });
              mediaStream.speaking = false;
              mediaStream.bargeIn = false;
              mediaStream.agentMessageFinished = true;
              mediaStream.logger.info('[IS_FINAL] Barge in value:', { bargeIn: mediaStream.bargeIn });
            }
            // Cancel ongoing LLM request ONLY if agent is not calling a tool
            if (!mediaStream.agentBusyCallingTool && mediaStream.requestId) {
              mediaStream.logger.info('[IS_FINAL] User interrupted assistant. Aborting LLM generation.', { cancelling_RequestId: mediaStream.requestId });
              mediaStream.validRequestIds.delete(mediaStream.requestId);
              mediaStream.requestId = null; // Reset requestId
              mediaStream.regularAssistantMessageID = null;
            }
            const sttEndTime = Date.now();
            mediaStream.logger.info(`Jobix STT process completed in: ${sttEndTime - mediaStream.sttStartTime} ms`);
            mediaStream.copyAllMessagesAiUser.push({
              role: 'stt',
              content: 'latency',
              timestamp: sttEndTime - mediaStream.sttStartTime
            });
            sttLatency.set({ call_id: mediaStream.callSid }, sttEndTime - mediaStream.sttStartTime);
            mediaStream.bargeIn = false;
            mediaStream.sttStartTime = null;
            mediaStream.logger.info('[IS_FINAL] Barge in value:', { bargeIn: mediaStream.bargeIn });
            promptLLM(mediaStream, transcript);
          })();
        }
      };
  
      ws.onclose = () => {
        // Clear any remaining buffered transcripts on close
        if (mediaStream.transcriptBuffer && mediaStream.transcriptBuffer.length > 0) {
          mediaStream.logger.info('Jobix STT: WebSocket closed with buffered transcripts', {
            lostBuffer: mediaStream.transcriptBuffer
          });
          mediaStream.transcriptBuffer = [];
        }
        mediaStream.logger.info("Jobix STT: WebSocket closed");
      };
    });
  };