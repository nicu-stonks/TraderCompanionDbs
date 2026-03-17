import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlarmSettings } from '../types';
import { priceAlertsAPI } from '../services/priceAlertsAPI';
import { Play, Pause } from 'lucide-react';
import { API_CONFIG } from '@/config';

interface AlarmSettingsFormProps {
  settings: AlarmSettings;
  onUpdate: (settings: Partial<AlarmSettings>) => Promise<void>;
}

export const AlarmSettingsForm: React.FC<AlarmSettingsFormProps> = ({ settings, onUpdate }) => {
  const [playDuration, setPlayDuration] = useState(settings.play_duration.toString());
  const [pauseDuration, setPauseDuration] = useState(settings.pause_duration.toString());
  const [cycles, setCycles] = useState(settings.cycles.toString());
  const [alarmSound, setAlarmSound] = useState(settings.alarm_sound_path);
  const [availableSounds, setAvailableSounds] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const audioHandlersRef = useRef<{ ended?: () => void; error?: (e: Event) => void }>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const cleanupAudio = (audio: HTMLAudioElement | null) => {
    if (!audio) return;

    // Remove event listeners using stored references
    if (audioHandlersRef.current.ended) {
      audio.removeEventListener('ended', audioHandlersRef.current.ended);
    }
    if (audioHandlersRef.current.error) {
      audio.removeEventListener('error', audioHandlersRef.current.error);
    }

    // Clear handlers
    audioHandlersRef.current = {};

    // Pause and clear src
    try {
      audio.pause();
      audio.src = '';
      audio.load(); // Reset the audio element
    } catch (e) {
      // Ignore errors during cleanup
    }
  };

  useEffect(() => {
    loadAvailableSounds();
  }, []);

  useEffect(() => {
    // Cleanup audio on unmount or when sound changes
    return () => {
      if (audioElement) {
        cleanupAudio(audioElement);
        setAudioElement(null);
        setIsPlaying(false);
      }
    };
  }, [audioElement]);

  const loadAvailableSounds = async () => {
    try {
      const response = await priceAlertsAPI.listAlarmSounds();
      setAvailableSounds(response.data.sounds);
    } catch (error) {
      console.error('Error loading sounds:', error);
      setAvailableSounds([settings.alarm_sound_path]);
    }
  };

  const handleSave = async () => {
    await onUpdate({
      play_duration: parseInt(playDuration),
      pause_duration: parseInt(pauseDuration),
      cycles: parseInt(cycles),
      alarm_sound_path: alarmSound,
    });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const response = await priceAlertsAPI.uploadAlarmSound(file);
      setAlarmSound(response.data.filename);
      // Reload all sounds
      await loadAvailableSounds();
      await onUpdate({ alarm_sound_path: response.data.filename });
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Failed to upload alarm sound. Please try again.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handlePlayPauseSound = () => {
    if (isPlaying && audioElement) {
      // Pause - properly clean up
      cleanupAudio(audioElement);
      setIsPlaying(false);
      setAudioElement(null);
    } else {
      // Clean up any existing audio first
      if (audioElement) {
        cleanupAudio(audioElement);
        setAudioElement(null);
      }

      // Play - use API endpoint for serving audio files
      const audioUrl = `${API_CONFIG.baseURL}/price_alerts/alarm-sounds/${encodeURIComponent(alarmSound)}`;
      // Create audio element with URL directly in constructor
      const audio = new Audio(audioUrl);
      audio.preload = 'auto';

      const handleEnded = () => {
        // Reset playing state when audio ends
        setIsPlaying(false);
        setAudioElement(null);
        cleanupAudio(audio);
      };

      const handleError = (err: Event) => {
        // Only handle if this audio element is still the current one and not being cleaned up
        if (audioElement !== audio || !audio.src) {
          return; // Ignore errors from cleaned-up audio elements
        }

        console.error('Error playing test sound:', err);
        console.error('Audio URL:', audioUrl);
        if (audio.currentSrc) {
          console.error('Audio currentSrc:', audio.currentSrc);
        }
        if (audio.error) {
          console.error('Audio error details:', {
            code: audio.error.code,
            message: audio.error.message,
          });
          alert(`Could not play sound: ${audio.error.message || 'Unknown error'}\n\nFile: ${alarmSound}\nURL: ${audioUrl}\n\nCheck browser console for details.`);
        }
        setIsPlaying(false);
        cleanupAudio(audio);
        setAudioElement(null);
      };

      // Store handler references
      audioHandlersRef.current.ended = handleEnded;
      audioHandlersRef.current.error = handleError;

      audio.addEventListener('ended', handleEnded);
      audio.addEventListener('error', handleError);

      audio.addEventListener('loadstart', () => {
        console.log('Loading audio:', audioUrl);
      });

      audio.addEventListener('canplay', () => {
        console.log('Audio can play:', audioUrl);
      });

      audio.addEventListener('loadeddata', () => {
        console.log('Audio data loaded:', audioUrl);
      });

      // Wait for audio to be ready before playing
      let playAttempted = false;
      const playAudio = () => {
        if (playAttempted) return;

        if (audio.readyState >= 2) { // HAVE_CURRENT_DATA or higher
          playAttempted = true;
          audio.play().then(() => {
            setIsPlaying(true);
            setAudioElement(audio);
          }).catch(err => {
            console.error('Error playing test sound:', err);
            console.error('Audio URL:', audioUrl);
            alert(`Could not play sound: ${err.message || 'Unknown error'}\n\nFile: ${alarmSound}\nURL: ${audioUrl}`);
            setIsPlaying(false);
            setAudioElement(null);
          });
        } else {
          // Wait a bit and try again
          setTimeout(() => {
            if (audio.readyState >= 2 && !playAttempted) {
              playAudio();
            } else if (!playAttempted) {
              handleError(new Event('timeout'));
            }
          }, 100);
        }
      };

      // Try to play when ready
      if (audio.readyState >= 2) {
        playAudio();
      } else {
        audio.addEventListener('canplay', playAudio, { once: true });
        // Fallback timeout
        setTimeout(() => {
          if (!playAttempted) {
            playAudio();
          }
        }, 2000);
      }
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Alarm Settings</CardTitle>
        <CardDescription>Configure how alerts are played</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="alarmSound">Alarm Sound</Label>
          <div className="flex gap-2">
            <Select value={alarmSound} onValueChange={setAlarmSound}>
              <SelectTrigger>
                <SelectValue placeholder="Select alarm sound" />
              </SelectTrigger>
              <SelectContent>
                {availableSounds.map((sound) => (
                  <SelectItem key={sound} value={sound}>
                    {sound}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              type="button"
              variant="outline"
              onClick={handlePlayPauseSound}
              title={isPlaying ? "Pause preview" : "Play preview"}
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="uploadSound">Upload Custom Alarm Sound</Label>
          <div className="flex gap-2">
            <Input
              ref={fileInputRef}
              id="uploadSound"
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="flex-1"
            />
            {isUploading && <span className="text-sm text-muted-foreground">Uploading...</span>}
          </div>
          <p className="text-xs text-muted-foreground">
            Supported formats: MP3, WAV, OGG, M4A
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="playDuration">Play Duration (seconds)</Label>
          <Input
            id="playDuration"
            type="number"
            min="1"
            value={playDuration}
            onChange={(e) => setPlayDuration(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="pauseDuration">Pause Duration (seconds)</Label>
          <Input
            id="pauseDuration"
            type="number"
            min="0"
            value={pauseDuration}
            onChange={(e) => setPauseDuration(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="cycles">Number of Cycles</Label>
          <Input
            id="cycles"
            type="number"
            min="1"
            value={cycles}
            onChange={(e) => setCycles(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            How many times to repeat the play-pause cycle
          </p>
        </div>

        <Button onClick={handleSave} className="w-full">
          Save Settings
        </Button>
      </CardContent>
    </Card>
  );
};

