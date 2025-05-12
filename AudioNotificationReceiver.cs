// File: AudioNotificationReceiver.cs
using UnityEngine;
using NativeWebSocket;
using System.Collections.Generic;

public class AudioNotificationReceiver : MonoBehaviour
{
    public string serverUrl = "ws://localhost:12347";
    public AudioSource audioSource;
    public int sampleRate = 24000;

    private WebSocket websocket;
    private Queue<float> audioQueue = new Queue<float>();

    private float[] audioBuffer;
    private int writePosition = 0;
    private int playPosition = 0;
    private bool isReceivingAudio = false;
    private bool hasStartedPlaying = false;

    private int bufferSamples = 24000 * 30; // 30 sec buffer

    async void Start()
    {
        audioBuffer = new float[bufferSamples];

        websocket = new WebSocket(serverUrl);

        websocket.OnOpen += () =>
        {
            Debug.Log("[AudioReceiver] Connected to server!");
            audioSource.clip = AudioClip.Create("StreamingClip", bufferSamples, 1, sampleRate, false);
            audioSource.loop = true;
            audioSource.Play();
        };

        websocket.OnMessage += OnAudioChunkReceived;
        websocket.OnError += (e) => Debug.LogError("WebSocket Error: " + e);
        websocket.OnClose += (e) => Debug.LogWarning("WebSocket Closed: " + e);

        await websocket.Connect();
    }

    void Update()
    {
        if (websocket != null)
        {
            websocket.DispatchMessageQueue();
        }

        if (audioQueue.Count > 0 && audioSource.clip != null)
        {
            WriteBufferedAudio();
            
            if (!audioSource.isPlaying)
            {
                audioSource.Play(); // 🚀 Resume automatically only if stopped
            }

            isReceivingAudio = true;
            hasStartedPlaying = true;
        }

        if (hasStartedPlaying)
        {
            float currentTimeSamples = audioSource.timeSamples;
            if (currentTimeSamples < playPosition) // Loop wrapped
            {
                playPosition = 0;
            }
            playPosition = (int)currentTimeSamples;
        }

        if (hasStartedPlaying && !isReceivingAudio && playPosition >= writePosition)
        {
            Debug.Log("[AudioReceiver] Audio finished naturally, stopping.");
            StopPlayback();
        }

        isReceivingAudio = false; // Reset each frame
    }
    private void OnAudioChunkReceived(byte[] bytes)
    {
        short[] samples = new short[bytes.Length / 2];
        System.Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);

        foreach (short s in samples)
        {
            audioQueue.Enqueue(s / 32768.0f);
        }
    }

    private void WriteBufferedAudio()
    {
        int samplesToWrite = Mathf.Min(audioQueue.Count, 2400); // 100ms chunks

        for (int i = 0; i < samplesToWrite; i++)
        {
            audioBuffer[writePosition] = audioQueue.Dequeue();
            writePosition++;

            if (writePosition >= bufferSamples)
            {
                writePosition = 0; // Wrap around
            }
        }

        audioSource.clip.SetData(audioBuffer, 0); // Write updated audio
    }

    private void StopPlayback()
    {
        // 🛑 Pause the AudioSource without destroying it
        audioSource.Stop();

        // 🧹 Clear the buffer safely
        System.Array.Clear(audioBuffer, 0, audioBuffer.Length);
        writePosition = 0;
        hasStartedPlaying = false;
        // 🔄 Reset sample data (wipe out old sound)
        if (audioSource.clip != null)
        {
            audioSource.clip.SetData(audioBuffer, 0);
        }

        Debug.Log("[AudioReceiver] Audio buffer cleared and playback paused.");
    }

    private async void OnApplicationQuit()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }
}
