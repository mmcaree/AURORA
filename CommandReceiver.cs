using UnityEngine;
using NativeWebSocket;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;

public class WebSocketBehaviorReceiver : MonoBehaviour {
    public string serverUrl = "ws://localhost:12348";
    private WebSocket websocket;
    private Queue<string> messageQueue = new Queue<string>();

    async void Start() {
        websocket = new WebSocket(serverUrl);

        websocket.OnOpen += () =>
        {
            Debug.Log("[BehaviorReceiver] Connected to server!");
        };

        websocket.OnMessage += (bytes) => {
            string jsonStr = Encoding.UTF8.GetString(bytes);
            lock (messageQueue) {
                messageQueue.Enqueue(jsonStr);
            }
        };

        websocket.OnError += (e) => Debug.LogError("BehaviorReceiver WebSocket Error: " + e);
        websocket.OnClose += (e) => Debug.LogWarning("BehaviorReceiver WebSocket Closed: " + e);

        await websocket.Connect();
    }

    void Update() {
        websocket?.DispatchMessageQueue();

        while (messageQueue.Count > 0) {
            string jsonStr;
            lock (messageQueue) {
                jsonStr = messageQueue.Dequeue();
            }

            Debug.Log("[BehaviorReceiver] Received: " + jsonStr);
            try {
                var behavior = JsonConvert.DeserializeObject<BehaviorCommand>(jsonStr);
                ApplyBehavior(behavior);
            } catch (System.Exception ex) {
                Debug.LogError("JSON Parse Error: " + ex.Message);
            }
        }
    }

    private void OnApplicationQuit() {
        websocket?.Close();
    }

    void ApplyBehavior(BehaviorCommand cmd) {
        Debug.Log($"Applying Intent: {cmd.intent}");
        // TODO: Apply to Animator or locomotion controller
    }

    public class BehaviorCommand {
        public string intent;
        public Locomotion locomotion;
        public Gesture gesture;

        public class Locomotion {
            public float VelocityX;
            public float VelocityY;
            public float StandStyle;
            public int LocomotionMode;
        }

        public class Gesture {
            public float ArmBlendH;
            public float ArmBlendV;
        }
    }
}
