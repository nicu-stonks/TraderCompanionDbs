"""
Standalone alarm player process - no Django dependencies.
This is spawned as a separate process and can be killed immediately.
"""
import os
import sys
import time
import pygame
from pathlib import Path


def play_alarm_standalone(sound_path, play_duration, pause_duration, cycles, stop_file):
    """
    Play alarm sound in a loop.
    This function runs in a separate process with no Django dependencies.
    Checks for stop_file existence to terminate gracefully.
    """
    try:
        print(f"[ALARM PROCESS] Starting with sound: {sound_path}", flush=True)
        print(f"[ALARM PROCESS] Stop file: {stop_file}", flush=True)
        print(f"[ALARM PROCESS] PID: {os.getpid()}", flush=True)
        
        pygame.mixer.init()
        pygame.mixer.music.load(sound_path)
        
        for cycle in range(cycles):
            # Check stop signal
            if Path(stop_file).exists():
                print("[ALARM PROCESS] Stop signal detected, exiting", flush=True)
                break
                
            print(f"[ALARM PROCESS] Playing cycle {cycle + 1}/{cycles}", flush=True)
            # Play with infinite loop (-1) so it repeats for the full duration
            pygame.mixer.music.play(-1)
            
            # Wait for the music to actually start playing
            while not pygame.mixer.music.get_busy():
                if Path(stop_file).exists():
                    print("[ALARM PROCESS] Stop signal before music started, exiting", flush=True)
                    pygame.mixer.music.stop()
                    return
                time.sleep(0.01)
            
            print(f"[ALARM PROCESS] Music is playing (looping for {play_duration}s)...", flush=True)
            
            # Wait for play duration, checking stop signal frequently
            start_time = time.time()
            while time.time() - start_time < play_duration:
                if Path(stop_file).exists():
                    print("[ALARM PROCESS] Stop signal during playback, exiting", flush=True)
                    pygame.mixer.music.stop()
                    return
                time.sleep(0.05)
            
            pygame.mixer.music.stop()
            print(f"[ALARM PROCESS] Stopped music for cycle {cycle + 1}", flush=True)
            
            # Check stop signal before pause
            if Path(stop_file).exists():
                print("[ALARM PROCESS] Stop signal before pause, exiting", flush=True)
                break
            
            # Pause between cycles
            if cycle < cycles - 1:
                print(f"[ALARM PROCESS] Pausing for {pause_duration}s", flush=True)
                pause_end = time.time() + pause_duration
                while time.time() < pause_end:
                    if Path(stop_file).exists():
                        print("[ALARM PROCESS] Stop signal during pause, exiting", flush=True)
                        return
                    time.sleep(0.05)
        
        print("[ALARM PROCESS] Finished all cycles", flush=True)
                    
    except Exception as e:
        print(f"[ALARM PROCESS] Error: {e}", flush=True)
    finally:
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
        print("[ALARM PROCESS] Cleanup complete, exiting", flush=True)


if __name__ == "__main__":
    # This allows the script to be run directly as a subprocess
    print(f"[ALARM PROCESS] Script started, args: {sys.argv}", flush=True)
    print(f"[ALARM PROCESS] Arg count: {len(sys.argv)}", flush=True)
    
    try:
        if len(sys.argv) == 6:
            sound_path = sys.argv[1]
            play_duration = float(sys.argv[2])
            pause_duration = float(sys.argv[3])
            cycles = int(sys.argv[4])
            stop_file = sys.argv[5]
            play_alarm_standalone(sound_path, play_duration, pause_duration, cycles, stop_file)
        else:
            print(f"[ALARM PROCESS] ERROR: Expected 6 arguments, got {len(sys.argv)}", flush=True)
            print(f"[ALARM PROCESS] Arguments: {sys.argv}", flush=True)
    except Exception as e:
        print(f"[ALARM PROCESS] FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

