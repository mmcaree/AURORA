# file: websocket_server.py
import asyncio
import websockets
import signal
import sys


behavior_clients = set()
tone_clients = set()
notification_clients = set()

async def behavior_handler(websocket):
    behavior_clients.add(websocket)
    print("[✅ Behavior Client Connected]")
    try:
        async for message in websocket:
            print(f"🧠 Received phi2 output: {message}")
            for client in behavior_clients:
                if client != websocket:
                    await websocket.send(message)
    except websockets.exceptions.ConnectionClosed:
        print("[⚡ Behavior Client Disconnected]")
    finally:
        behavior_clients.remove(websocket)

async def tone_handler(websocket):
    tone_clients.add(websocket)
    print("[✅ Tone Client Connected]")
    try:
        async for message in websocket:
            #print(f"[🎭 Tone]")
            for client in tone_clients:
                if client != websocket:
                    await client.send(message)
    except websockets.exceptions.ConnectionClosed:
        print("[⚡ Tone Client Disconnected]")
    finally:
        tone_clients.remove(websocket)

async def notification_handler(websocket):
    notification_clients.add(websocket)
    print("[✅ Notification Client Connected]")
    try:
        async for message in websocket:
            #print(f"[🎭 Tone]")
            for client in notification_clients:
                if client != websocket:
                    await client.send(message)
    except websockets.exceptions.ConnectionClosed:
        print("[⚡ Notification Client Disconnected]")
    finally:
        notification_clients.remove(websocket)

async def main():
    behavior_server = await websockets.serve(behavior_handler, "localhost", 12348)
    tone_server = await websockets.serve(tone_handler, "localhost", 12346)
    notification_server = await websockets.serve(notification_handler, "localhost", 12347)

    print("[✅ Server Running]")
    print("Phoneme WebSocket: ws://localhost:12348")
    print("Tone WebSocket:    ws://localhost:12346")
    print("Notification WebSocket:    ws://localhost:12347")

    # Wait for manual Ctrl+C
    await asyncio.Future()  # run forever

def shutdown(signal_received, frame):
    print("\n[🛑 Ctrl+C Received] Shutting down cleanly...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup Ctrl+C handler
    signal.signal(signal.SIGINT, shutdown)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        shutdown(None, None)
