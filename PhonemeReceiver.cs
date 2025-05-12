// File: PhonemeReceiver.cs
using UnityEngine;
using NativeWebSocket;
using System.Collections.Generic;

public class PhonemeReceiver : MonoBehaviour
{
    public SkinnedMeshRenderer faceMesh;
    private WebSocket websocket;

    private Dictionary<string, int> phonemeToBlendshape = new Dictionary<string, int>();

    void Start()
    {
        websocket = new WebSocket("ws://localhost:12345");

        websocket.OnOpen += () => Debug.Log("WebSocket connected!");
        websocket.OnError += (e) => Debug.Log("WebSocket Error: " + e);
        websocket.OnClose += (e) => Debug.Log("WebSocket Closed");

        websocket.OnMessage += (bytes) =>
        {
            string phoneme = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log($"Received phoneme: {phoneme}");
            ApplyPhoneme(phoneme);
        };

        websocket.Connect();

        // Setup mappings
        SetupPhonemeMappings();
    }

    void SetupPhonemeMappings()
    {
        // Find blendshape indices dynamically
        phonemeToBlendshape.Add("a", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_aa"));
        phonemeToBlendshape.Add("e", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_ee"));
        phonemeToBlendshape.Add("i", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_ih"));
        phonemeToBlendshape.Add("o", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_oh"));
        phonemeToBlendshape.Add("u", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_ou"));
        phonemeToBlendshape.Add("f", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_ff"));
        phonemeToBlendshape.Add("θ", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_th"));
        phonemeToBlendshape.Add("d", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_dd"));
        phonemeToBlendshape.Add("k", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_kk"));
        phonemeToBlendshape.Add("tʃ", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_ch"));
        phonemeToBlendshape.Add("s", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_ss"));
        phonemeToBlendshape.Add("n", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_nn"));
        phonemeToBlendshape.Add("r", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_rr"));
        phonemeToBlendshape.Add("sil", faceMesh.sharedMesh.GetBlendShapeIndex("vrc.v_sil"));
    }

    private void ApplyPhoneme(string phoneme)
    {
        // Reset all visemes
        foreach (var index in phonemeToBlendshape.Values)
        {
            if (index >= 0)
                faceMesh.SetBlendShapeWeight(index, 0f);
        }

        // Apply matching viseme
        if (phonemeToBlendshape.TryGetValue(phoneme, out int visemeIndex))
        {
            if (visemeIndex >= 0)
                faceMesh.SetBlendShapeWeight(visemeIndex, 100f);
        }
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
