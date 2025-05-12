// File: ToneReceiver.cs
using UnityEngine;
using NativeWebSocket;

public class ToneReceiver : MonoBehaviour
{
    public AIFullBehaviorController aiController;
    private WebSocket websocket;

    async void Start()
    {
        websocket = new WebSocket("ws://localhost:12346");

        websocket.OnOpen += () => Debug.Log("Tone WebSocket Connected!");
        websocket.OnError += (e) => Debug.Log("Tone WebSocket Error: " + e);
        websocket.OnClose += (e) => Debug.Log("Tone WebSocket Closed");

        websocket.OnMessage += (bytes) =>
        {
            string tone = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log($"[Tone Received]: {tone}");
            //will pass tone once UpdateMood implemented
            aiController.UpdateMood(tone);
        };

        await websocket.Connect();
    }

    void Update()
    {
        websocket?.DispatchMessageQueue();
    }

    private void OnDestroy()
    {
        websocket?.Close();
    }
}
