"use client";

import { useEffect, useRef } from 'react';

export interface CosmicSounds {
  playBlockMined: () => void;
  playTransaction: () => void;
  playSearch: () => void;
  playError: () => void;
  playSuccess: () => void;
  playAmbient: () => void;
  stopAmbient: () => void;
}

export function useCosmicSounds(): CosmicSounds {
  const audioContextRef = useRef<AudioContext | null>(null);
  const ambientOscillatorRef = useRef<OscillatorNode | null>(null);
  
  useEffect(() => {
    // Initialize Web Audio API
    if (typeof window !== 'undefined') {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    return () => {
      if (ambientOscillatorRef.current) {
        ambientOscillatorRef.current.disconnect();
      }
    };
  }, []);

  const createBeep = (frequency: number, duration: number, type: OscillatorType = 'sine', volume = 0.1) => {
    if (!audioContextRef.current) return;
    
    const oscillator = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContextRef.current.destination);
    
    oscillator.frequency.value = frequency;
    oscillator.type = type;
    
    gainNode.gain.setValueAtTime(0, audioContextRef.current.currentTime);
    gainNode.gain.linearRampToValueAtTime(volume, audioContextRef.current.currentTime + 0.01);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContextRef.current.currentTime + duration);
    
    oscillator.start(audioContextRef.current.currentTime);
    oscillator.stop(audioContextRef.current.currentTime + duration);
  };

  const createChord = (frequencies: number[], duration: number, type: OscillatorType = 'sine', volume = 0.05) => {
    frequencies.forEach(freq => {
      createBeep(freq, duration, type, volume);
    });
  };

  return {
    playBlockMined: () => {
      // Cosmic block mining sound - ascending chord
      createChord([220, 330, 440, 550], 0.8, 'triangle', 0.08);
      setTimeout(() => createBeep(880, 0.3, 'square', 0.06), 200);
    },
    
    playTransaction: () => {
      // Quick transaction beep
      createBeep(660, 0.15, 'sine', 0.05);
      setTimeout(() => createBeep(880, 0.1, 'sine', 0.03), 100);
    },
    
    playSearch: () => {
      // Search scanning sound
      const startTime = audioContextRef.current?.currentTime || 0;
      for (let i = 0; i < 5; i++) {
        setTimeout(() => {
          createBeep(400 + (i * 100), 0.1, 'sawtooth', 0.03);
        }, i * 50);
      }
    },
    
    playError: () => {
      // Error sound - descending
      createBeep(400, 0.2, 'sawtooth', 0.1);
      setTimeout(() => createBeep(300, 0.3, 'sawtooth', 0.1), 150);
    },
    
    playSuccess: () => {
      // Success chime - ascending
      createBeep(523, 0.2, 'sine', 0.06); // C
      setTimeout(() => createBeep(659, 0.2, 'sine', 0.06), 100); // E
      setTimeout(() => createBeep(784, 0.3, 'sine', 0.06), 200); // G
    },
    
    playAmbient: () => {
      if (!audioContextRef.current || ambientOscillatorRef.current) return;
      
      // Cosmic ambient drone
      const oscillator1 = audioContextRef.current.createOscillator();
      const oscillator2 = audioContextRef.current.createOscillator();
      const gainNode = audioContextRef.current.createGain();
      
      oscillator1.connect(gainNode);
      oscillator2.connect(gainNode);
      gainNode.connect(audioContextRef.current.destination);
      
      oscillator1.frequency.value = 55; // Low A
      oscillator2.frequency.value = 110; // A one octave up
      oscillator1.type = 'sine';
      oscillator2.type = 'triangle';
      
      gainNode.gain.setValueAtTime(0, audioContextRef.current.currentTime);
      gainNode.gain.linearRampToValueAtTime(0.02, audioContextRef.current.currentTime + 2);
      
      oscillator1.start();
      oscillator2.start();
      
      ambientOscillatorRef.current = oscillator1;
    },
    
    stopAmbient: () => {
      if (ambientOscillatorRef.current && audioContextRef.current) {
        const gainNode = audioContextRef.current.createGain();
        gainNode.gain.linearRampToValueAtTime(0, audioContextRef.current.currentTime + 1);
        ambientOscillatorRef.current.stop(audioContextRef.current.currentTime + 1);
        ambientOscillatorRef.current = null;
      }
    }
  };
}