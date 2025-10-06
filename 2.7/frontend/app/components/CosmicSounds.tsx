"use client";

import { useEffect, useRef } from 'react';

interface CosmicSoundEffects {
  playSearch: () => void;
  playBlockFound: () => void;
  playTransactionConfirm: () => void;
  playCosmicAmbient: () => void;
  stopAmbient: () => void;
  playHashrateBoost: () => void;
  playMiningReward: () => void;
}

// Web Audio API based cosmic sound generator
export const useCosmicSounds = (): CosmicSoundEffects => {
  const audioContextRef = useRef<AudioContext | null>(null);
  const ambientOscillatorRef = useRef<OscillatorNode | null>(null);
  const ambientGainRef = useRef<GainNode | null>(null);

  useEffect(() => {
    // Initialize Web Audio Context
    if (typeof window !== 'undefined') {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    return () => {
      if (ambientOscillatorRef.current) {
        ambientOscillatorRef.current.stop();
      }
    };
  }, []);

  const createTone = (frequency: number, duration: number, type: OscillatorType = 'sine') => {
    if (!audioContextRef.current) return;

    const oscillator = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContextRef.current.destination);

    oscillator.frequency.setValueAtTime(frequency, audioContextRef.current.currentTime);
    oscillator.type = type;

    gainNode.gain.setValueAtTime(0.1, audioContextRef.current.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContextRef.current.currentTime + duration);

    oscillator.start(audioContextRef.current.currentTime);
    oscillator.stop(audioContextRef.current.currentTime + duration);
  };

  const createChord = (frequencies: number[], duration: number) => {
    frequencies.forEach((freq, index) => {
      setTimeout(() => createTone(freq, duration), index * 50);
    });
  };

  const playSearch = () => {
    // Cosmic search swoosh - ascending frequencies
    createTone(220, 0.2, 'square');
    setTimeout(() => createTone(330, 0.2, 'square'), 100);
    setTimeout(() => createTone(440, 0.2, 'square'), 200);
  };

  const playBlockFound = () => {
    // Divine block discovery - harmonic chord
    createChord([261.63, 329.63, 392.00, 523.25], 1.0); // C major chord
    
    // Add sparkle effect
    setTimeout(() => {
      for (let i = 0; i < 5; i++) {
        setTimeout(() => createTone(1000 + Math.random() * 1000, 0.1, 'triangle'), i * 50);
      }
    }, 200);
  };

  const playTransactionConfirm = () => {
    // Transaction confirmation - gentle bell
    createTone(880, 0.5, 'triangle');
    setTimeout(() => createTone(1760, 0.3, 'triangle'), 200);
  };

  const playCosmicAmbient = () => {
    if (!audioContextRef.current || ambientOscillatorRef.current) return;

    // Create ambient cosmic drone
    const oscillator1 = audioContextRef.current.createOscillator();
    const oscillator2 = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();

    oscillator1.connect(gainNode);
    oscillator2.connect(gainNode);
    gainNode.connect(audioContextRef.current.destination);

    oscillator1.frequency.setValueAtTime(110, audioContextRef.current.currentTime); // Deep bass
    oscillator2.frequency.setValueAtTime(220.5, audioContextRef.current.currentTime); // Slight detune for richness
    
    oscillator1.type = 'sine';
    oscillator2.type = 'triangle';

    gainNode.gain.setValueAtTime(0.03, audioContextRef.current.currentTime); // Very quiet ambient

    oscillator1.start();
    oscillator2.start();

    ambientOscillatorRef.current = oscillator1;
    ambientGainRef.current = gainNode;

    // Add slow frequency modulation for cosmic feel
    const lfo = audioContextRef.current.createOscillator();
    const lfoGain = audioContextRef.current.createGain();
    
    lfo.connect(lfoGain);
    lfoGain.connect(oscillator2.frequency);
    
    lfo.frequency.setValueAtTime(0.1, audioContextRef.current.currentTime); // 0.1 Hz modulation
    lfoGain.gain.setValueAtTime(5, audioContextRef.current.currentTime);
    
    lfo.start();
  };

  const stopAmbient = () => {
    if (ambientOscillatorRef.current && ambientGainRef.current) {
      ambientGainRef.current.gain.exponentialRampToValueAtTime(0.001, audioContextRef.current!.currentTime + 2);
      setTimeout(() => {
        if (ambientOscillatorRef.current) {
          ambientOscillatorRef.current.stop();
          ambientOscillatorRef.current = null;
        }
      }, 2000);
    }
  };

  const playHashrateBoost = () => {
    // Mining power up sound - rising synthesizer
    const baseFreq = 200;
    for (let i = 0; i < 10; i++) {
      setTimeout(() => {
        createTone(baseFreq * (1 + i * 0.1), 0.1, 'sawtooth');
      }, i * 50);
    }
  };

  const playMiningReward = () => {
    // Reward sound - golden coins falling
    const coinFreqs = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    coinFreqs.forEach((freq, index) => {
      setTimeout(() => {
        createTone(freq, 0.4, 'triangle');
        // Add harmonic
        setTimeout(() => createTone(freq * 2, 0.2, 'triangle'), 100);
      }, index * 150);
    });
  };

  return {
    playSearch,
    playBlockFound,
    playTransactionConfirm,
    playCosmicAmbient,
    stopAmbient,
    playHashrateBoost,
    playMiningReward
  };
};

export default useCosmicSounds;