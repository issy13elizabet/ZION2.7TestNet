#!/usr/bin/env python3
"""
ZION 2.6.75 Music AI Compositor
AI Music Composition, Harmonic Analysis, Emotional Resonance Modeling & NFT Music Assets
🌌 ON THE STAR - Sonic Consciousness Engine
"""

import asyncio
import math
import random
import time
import json
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import base64

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Optional MIDI export support
try:
    import mido  # type: ignore
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

# Optional simple ML (placeholder) - if sklearn not installed we fallback
try:
    from sklearn.feature_extraction import DictVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ScaleType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"


class Genre(Enum):
    AMBIENT = "ambient"
    LOFI = "lofi"
    TRANCE = "trance"
    CINEMATIC = "cinematic"
    ORCHESTRAL = "orchestral"
    SYNTHWAVE = "synthwave"
    JAZZ = "jazz"
    HIPHOP = "hiphop"
    METAL = "metal"
    EXPERIMENTAL = "experimental"


class Emotion(Enum):
    CALM = "calm"
    ENERGETIC = "energetic"
    MYSTICAL = "mystical"
    UPLIFTING = "uplifting"
    MELANCHOLIC = "melancholic"
    HEROIC = "heroic"
    DARK = "dark"
    SACRED = "sacred"
    TRANSCENDENT = "transcendent"
    PLAYFUL = "playful"


@dataclass
class EmotionProfile:
    target_emotions: List[Emotion]
    energy_level: float  # 0-1
    tension: float       # 0-1
    brightness: float    # 0-1 (major vs minor leaning)
    depth: float         # 0-1 (harmonic complexity)


@dataclass
class Note:
    pitch: int
    start: float  # beats
    duration: float  # beats
    velocity: int
    channel: int = 0


@dataclass
class Track:
    track_id: str
    name: str
    instrument: str
    notes: List[Note] = field(default_factory=list)
    effects: Dict[str, Any] = field(default_factory=dict)
    role: str = "unknown"  # melody, harmony, bass, percussion, texture


@dataclass
class Composition:
    composition_id: str
    title: str
    genre: Genre
    tempo: int
    scale_root: str
    scale_type: ScaleType
    duration_beats: float
    emotion_profile: EmotionProfile
    tracks: List[Track] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    harmonic_progression: List[str] = field(default_factory=list)
    key_signature: str = "C Major"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NFTMusicAsset:
    asset_id: str
    composition_id: str
    owner_address: str
    mint_tx: Optional[str]
    token_uri: Optional[str]
    created_at: float = field(default_factory=time.time)
    royalties: float = 0.05
    metadata_hash: Optional[str] = None


class ZionMusicAI:
    """ZION Music AI Compositor"""

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    SCALE_PATTERNS = {
        ScaleType.MAJOR:        [2,2,1,2,2,2,1],
        ScaleType.MINOR:        [2,1,2,2,1,2,2],
        ScaleType.DORIAN:       [2,1,2,2,2,1,2],
        ScaleType.PHRYGIAN:     [1,2,2,2,1,2,2],
        ScaleType.LYDIAN:       [2,2,2,1,2,2,1],
        ScaleType.MIXOLYDIAN:   [2,2,1,2,2,1,2],
        ScaleType.LOCRIAN:      [1,2,2,1,2,2,2],
        ScaleType.HARMONIC_MINOR:[2,1,2,2,1,3,1],
        ScaleType.MELODIC_MINOR: [2,1,2,2,2,2,1]
    }

    GENRE_DEFAULTS = {
        Genre.AMBIENT:      {'tempo': 70,  'instruments': ['pad','texture','bass'], 'swing':0.0, 'density':0.3},
        Genre.LOFI:         {'tempo': 82,  'instruments': ['piano','drums','bass','tape_noise'], 'swing':0.55,'density':0.5},
        Genre.TRANCE:       {'tempo': 138, 'instruments': ['saw_lead','supersaw','bass','drums'], 'swing':0.0,'density':0.8},
        Genre.CINEMATIC:    {'tempo': 110, 'instruments': ['strings','brass','perc','choir'], 'swing':0.0,'density':0.7},
        Genre.ORCHESTRAL:   {'tempo': 100, 'instruments': ['strings','woodwinds','brass','perc'], 'swing':0.0,'density':0.65},
        Genre.SYNTHWAVE:    {'tempo': 115, 'instruments': ['bass_synth','lead','pad','drums'], 'swing':0.0,'density':0.6},
        Genre.JAZZ:         {'tempo': 140, 'instruments': ['piano','upright_bass','ride','sax'], 'swing':0.62,'density':0.7},
        Genre.HIPHOP:       {'tempo': 92,  'instruments': ['drums','bass','keys','vox'], 'swing':0.58,'density':0.5},
        Genre.METAL:        {'tempo': 160, 'instruments': ['dist_gtr','drums','bass','lead'], 'swing':0.0,'density':0.9},
        Genre.EXPERIMENTAL: {'tempo': 123, 'instruments': ['modular','glitch','texture','perc'], 'swing':0.0,'density':0.75}
    }

    HARMONIC_FUNCTIONS = ["I","ii","iii","IV","V","vi","vii°"]
    COMMON_PROGRESSIONS = [
        ["I","V","vi","IV"],
        ["ii","V","I","vi"],
        ["I","vi","IV","V"],
        ["I","IV","V","IV"],
        ["I","IV","vi","V"],
        ["vi","IV","I","V"],
        ["I","V","IV","V"],
        ["I","III","IV","iv"],
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.compositions: Dict[str, Composition] = {}
        self.nft_assets: Dict[str, NFTMusicAsset] = {}

        # Simple genre classifier (placeholder)
        self.genre_classifier = None
        if SKLEARN_AVAILABLE:
            self._initialize_genre_classifier()

        self.logger.info("🎵 ZionMusicAI initialized")

    # ------------------------- GENRE CLASSIFIER ---------------------------
    def _initialize_genre_classifier(self):
        try:
            # Minimal dummy classifier with handcrafted features
            self.vectorizer = DictVectorizer()
            self.genre_classifier = LogisticRegression(max_iter=200)
            train_samples = [
                ({'tempo': 70, 'density':0.3,'swing':0.0,'energy':0.2}, Genre.AMBIENT.value),
                ({'tempo': 82, 'density':0.5,'swing':0.55,'energy':0.4}, Genre.LOFI.value),
                ({'tempo': 138,'density':0.8,'swing':0.0,'energy':0.9}, Genre.TRANCE.value),
                ({'tempo': 110,'density':0.7,'swing':0.0,'energy':0.7}, Genre.CINEMATIC.value),
                ({'tempo': 140,'density':0.7,'swing':0.62,'energy':0.6}, Genre.JAZZ.value),
            ]
            X_dict = [s[0] for s in train_samples]
            y = [s[1] for s in train_samples]
            X = self.vectorizer.fit_transform(X_dict)
            self.genre_classifier.fit(X, y)
        except Exception as e:
            self.logger.warning(f"Genre classifier init failed: {e}")
            self.genre_classifier = None

    def classify_genre(self, features: Dict[str, float]) -> Optional[Genre]:
        try:
            if self.genre_classifier:
                X = self.vectorizer.transform([features])
                pred = self.genre_classifier.predict(X)[0]
                return Genre(pred)
        except Exception as e:
            self.logger.warning(f"Genre classification failed: {e}")
        return None

    # -------------------------- SCALE & HARMONY ---------------------------
    def build_scale(self, root: str, scale_type: ScaleType) -> List[int]:
        root_index = self.NOTE_NAMES.index(root)
        pattern = self.SCALE_PATTERNS[scale_type]
        notes = [root_index]
        current = root_index
        for step in pattern[:-1]:
            current = (current + step) % 12
            notes.append(current)
        return notes

    def generate_progression(self, scale_type: ScaleType, length: int, emotion_profile: EmotionProfile) -> List[str]:
        prog = random.choice(self.COMMON_PROGRESSIONS).copy()
        # Adjust for emotion: if melancholic or dark prefer vi / ii / iv heavy
        if any(e in [Emotion.MELANCHOLIC, Emotion.DARK] for e in emotion_profile.target_emotions):
            if random.random() < 0.6:
                prog = ["vi" if c=="I" else c for c in prog]
        # Tension influences inclusion of V and secondary dominants (simplified)
        if emotion_profile.tension > 0.6 and random.random() < 0.5:
            prog.append("V")
        # Extend to requested length via repetition / variation
        out = []
        while len(out) < length:
            var = prog.copy()
            if random.random() < 0.3:
                random.shuffle(var)
            out.extend(var)
        return out[:length]

    # -------------------------- COMPOSITION CORE --------------------------
    async def compose_music(self,
                            title: str,
                            genre: Genre,
                            emotion_profile: Optional[EmotionProfile] = None,
                            duration_bars: int = 16,
                            scale_root: str = 'C',
                            scale_type: ScaleType = ScaleType.MINOR,
                            tempo: Optional[int] = None) -> Dict[str, Any]:
        """Main high-level composition function."""
        try:
            comp_id = self._generate_id(prefix="cmp")
            genre_defaults = self.GENRE_DEFAULTS[genre]
            tempo = tempo or genre_defaults['tempo']
            if not emotion_profile:
                emotion_profile = EmotionProfile(
                    target_emotions=[Emotion.CALM] if genre==Genre.AMBIENT else [Emotion.ENERGETIC],
                    energy_level=0.3 if genre==Genre.AMBIENT else 0.7,
                    tension=0.4,
                    brightness=0.6,
                    depth=0.5
                )

            beats_per_bar = 4
            duration_beats = duration_bars * beats_per_bar

            progression = self.generate_progression(scale_type, max(4, duration_bars//4), emotion_profile)
            key_signature = f"{scale_root} {scale_type.value.replace('_',' ').title()}"

            composition = Composition(
                composition_id=comp_id,
                title=title,
                genre=genre,
                tempo=tempo,
                scale_root=scale_root,
                scale_type=scale_type,
                duration_beats=duration_beats,
                emotion_profile=emotion_profile,
                harmonic_progression=progression,
                key_signature=key_signature,
                metadata={
                    'swing': genre_defaults['swing'],
                    'density': genre_defaults['density'],
                    'created_at_iso': datetime.utcnow().isoformat()
                }
            )

            # Generate core tracks
            scale_notes = self.build_scale(scale_root, scale_type)
            composition.tracks.append(self._create_chord_track(composition, scale_notes))
            composition.tracks.append(self._create_melody_track(composition, scale_notes, emotion_profile))
            composition.tracks.append(self._create_bass_track(composition, scale_notes))
            if genre in [Genre.HIPHOP, Genre.TRANCE, Genre.METAL, Genre.SYNTHWAVE, Genre.JAZZ]:
                composition.tracks.append(self._create_drum_track(composition))
            # Texture / pad
            if genre in [Genre.AMBIENT, Genre.CINEMATIC, Genre.ORCHESTRAL, Genre.SYNTHWAVE, Genre.EXPERIMENTAL]:
                composition.tracks.append(self._create_texture_track(composition, scale_notes))

            self.compositions[comp_id] = composition
            self.logger.info(f"🎶 Composition created: {title} ({genre.value})")
            return {'success': True, 'composition_id': comp_id, 'tracks': len(composition.tracks)}
        except Exception as e:
            self.logger.error(f"Composition failed: {e}")
            return {'success': False, 'error': str(e)}

    def _create_chord_track(self, composition: Composition, scale_notes: List[int]) -> Track:
        track = Track(track_id=self._generate_id('trk'), name='Chords', instrument='pad', role='harmony')
        beats = 0.0
        chord_len = 4.0  # whole note per chord
        progression_cycle = composition.harmonic_progression
        idx = 0
        while beats < composition.duration_beats:
            func = progression_cycle[idx % len(progression_cycle)]
            chord_pitches = self._chord_from_function(func, scale_notes, base_octave=4)
            for p in chord_pitches:
                track.notes.append(Note(pitch=p, start=beats, duration=chord_len, velocity=70, channel=1))
            beats += chord_len
            idx += 1
        return track

    def _create_melody_track(self, composition: Composition, scale_notes: List[int], emotion: EmotionProfile) -> Track:
        track = Track(track_id=self._generate_id('trk'), name='Melody', instrument='lead', role='melody')
        density = composition.metadata['density']
        time_pos = 0.0
        beat_step = 0.5  # eighth notes
        while time_pos < composition.duration_beats:
            if random.random() < density:
                pitch = self._select_melodic_pitch(scale_notes, base_octave=5, emotion=emotion)
                dur = random.choice([0.25,0.5,1.0])
                track.notes.append(Note(pitch=pitch, start=time_pos, duration=dur, velocity=80, channel=0))
            time_pos += beat_step
        return track

    def _create_bass_track(self, composition: Composition, scale_notes: List[int]) -> Track:
        track = Track(track_id=self._generate_id('trk'), name='Bass', instrument='bass', role='bass')
        beats = 0.0
        step = 1.0
        prog_idx = 0
        while beats < composition.duration_beats:
            func = composition.harmonic_progression[prog_idx % len(composition.harmonic_progression)]
            root_pitch = self._chord_root_from_function(func, scale_notes, base_octave=2)
            dur = 1.0
            track.notes.append(Note(pitch=root_pitch, start=beats, duration=dur, velocity=90, channel=2))
            beats += step
            prog_idx += 1
        return track

    def _create_drum_track(self, composition: Composition) -> Track:
        track = Track(track_id=self._generate_id('trk'), name='Drums', instrument='drum_kit', role='percussion')
        beats = 0.0
        while beats < composition.duration_beats:
            # Kick
            track.notes.append(Note(pitch=36, start=beats, duration=0.25, velocity=100, channel=9))
            # Snare on 2 & 4
            if int(beats) % 4 == 2:
                track.notes.append(Note(pitch=38, start=beats, duration=0.25, velocity=110, channel=9))
            # Hi-hat pattern
            if random.random() < 0.8:
                track.notes.append(Note(pitch=42, start=beats+0.5, duration=0.25, velocity=60, channel=9))
            beats += 1.0
        return track

    def _create_texture_track(self, composition: Composition, scale_notes: List[int]) -> Track:
        track = Track(track_id=self._generate_id('trk'), name='Texture', instrument='atmosphere', role='texture')
        time_pos = 0.0
        while time_pos < composition.duration_beats:
            pitch = random.choice(scale_notes) + 12*6
            track.notes.append(Note(pitch=pitch, start=time_pos, duration=4.0, velocity=40, channel=3))
            time_pos += 4.0
        return track

    # ----------------------- HARMONIC UTILITIES ---------------------------
    def _chord_from_function(self, func: str, scale_notes: List[int], base_octave: int = 4) -> List[int]:
        degree_map = {"I":0,"ii":1,"iii":2,"IV":3,"V":4,"vi":5,"vii°":6,"III":2,"iv":3}
        degree = degree_map.get(func,0)
        root = scale_notes[degree] + base_octave*12
        third = scale_notes[(degree+2)%7] + base_octave*12
        fifth = scale_notes[(degree+4)%7] + base_octave*12
        return [root, third, fifth]

    def _chord_root_from_function(self, func: str, scale_notes: List[int], base_octave: int = 2) -> int:
        degree_map = {"I":0,"ii":1,"iii":2,"IV":3,"V":4,"vi":5,"vii°":6,"III":2,"iv":3}
        degree = degree_map.get(func,0)
        return scale_notes[degree] + base_octave*12

    def _select_melodic_pitch(self, scale_notes: List[int], base_octave: int, emotion: EmotionProfile) -> int:
        # Brightness biases toward upper scale degrees
        idx = random.choices(range(len(scale_notes)), weights=[1+(i*emotion.brightness) for i in range(len(scale_notes))])[0]
        return scale_notes[idx] + base_octave*12

    # ----------------------- ANALYSIS & EXPORT ---------------------------
    async def analyze_harmony(self, composition_id: str) -> Dict[str, Any]:
        if composition_id not in self.compositions:
            return {'success': False, 'error': 'Composition not found'}
        comp = self.compositions[composition_id]
        chord_counts = {}
        for f in comp.harmonic_progression:
            chord_counts[f] = chord_counts.get(f,0)+1
        diversity = len(chord_counts)/max(len(comp.harmonic_progression),1)
        return {
            'success': True,
            'composition_id': composition_id,
            'chord_distribution': chord_counts,
            'diversity_index': diversity
        }

    async def export_composition(self, composition_id: str, format: str = 'json') -> Dict[str, Any]:
        if composition_id not in self.compositions:
            return {'success': False, 'error': 'Composition not found'}
        comp = self.compositions[composition_id]
        if format == 'json':
            data = asdict(comp)
            return {'success': True, 'format': 'json', 'data': data}
        elif format == 'midi':
            if not MIDO_AVAILABLE:
                return {'success': False, 'error': 'mido not installed'}
            try:
                mid = mido.MidiFile()
                ticks_per_beat = 480
                mid.ticks_per_beat = ticks_per_beat
                for tr in comp.tracks:
                    mtrack = mido.MidiTrack()
                    mid.tracks.append(mtrack)
                    last_tick = 0
                    for note in sorted(tr.notes, key=lambda n: n.start):
                        start_tick = int(note.start * ticks_per_beat)
                        delta = start_tick - last_tick
                        mtrack.append(mido.Message('note_on', note=note.pitch, velocity=note.velocity, time=delta, channel=note.channel))
                        end_tick = start_tick + int(note.duration * ticks_per_beat)
                        mtrack.append(mido.Message('note_off', note=note.pitch, velocity=0, time=end_tick-start_tick, channel=note.channel))
                        last_tick = end_tick
                # Encode MIDI in base64
                import io
                buf = io.BytesIO()
                mid.save(file=buf)
                b64 = base64.b64encode(buf.getvalue()).decode()
                return {'success': True, 'format': 'midi_base64', 'data': b64}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': 'Unsupported format'}

    # ----------------------- NFT MINTING (STUB) --------------------------
    async def mint_music_nft(self, composition_id: str, owner_address: str) -> Dict[str, Any]:
        if composition_id not in self.compositions:
            return {'success': False, 'error': 'Composition not found'}
        comp = self.compositions[composition_id]
        # Create metadata hash
        comp_json = json.dumps(asdict(comp), sort_keys=True)
        meta_hash = hashlib.sha256(comp_json.encode()).hexdigest()
        asset_id = self._generate_id('nft')
        # Stub transaction id
        tx_id = hashlib.md5(f"{asset_id}{meta_hash}".encode()).hexdigest()
        nft = NFTMusicAsset(
            asset_id=asset_id,
            composition_id=composition_id,
            owner_address=owner_address,
            mint_tx=tx_id,
            token_uri=f"ipfs://placeholder/{meta_hash}",
            metadata_hash=meta_hash,
            royalties=0.05
        )
        self.nft_assets[asset_id] = nft
        return {
            'success': True,
            'asset_id': asset_id,
            'mint_tx': tx_id,
            'metadata_hash': meta_hash
        }

    # ----------------------- ANALYTICS -----------------------------------
    async def get_music_analytics(self) -> Dict[str, Any]:
        try:
            total = len(self.compositions)
            genres = {}
            avg_tempo = 0
            total_notes = 0
            for comp in self.compositions.values():
                genres[comp.genre.value] = genres.get(comp.genre.value,0)+1
                avg_tempo += comp.tempo
                for tr in comp.tracks:
                    total_notes += len(tr.notes)
            if total>0:
                avg_tempo /= total
            return {
                'success': True,
                'total_compositions': total,
                'genre_distribution': genres,
                'average_tempo': avg_tempo,
                'total_notes': total_notes,
                'nft_assets': len(self.nft_assets)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ----------------------- UTILITY -------------------------------------
    def _generate_id(self, prefix: str) -> str:
        rnd = random.getrandbits(48)
        return f"{prefix}_{rnd:012x}"

    # ----------------------- DEMO ----------------------------------------
    async def demo(self):
        print("🎵 ZION Music AI Demo")
        emotion = EmotionProfile(
            target_emotions=[Emotion.CALM, Emotion.TRANSCENDENT],
            energy_level=0.35,
            tension=0.3,
            brightness=0.55,
            depth=0.6
        )
        result = await self.compose_music(
            title="Celestial Drift",
            genre=Genre.AMBIENT,
            emotion_profile=emotion,
            duration_bars=12,
            scale_root='C',
            scale_type=ScaleType.DORIAN
        )
        print("Compose:", result)
        if result['success']:
            cid = result['composition_id']
            harmony = await self.analyze_harmony(cid)
            print("Harmony:", harmony)
            export = await self.export_composition(cid)
            print("Export sample keys:", list(export['data'].keys()) if export['success'] else export)
            nft = await self.mint_music_nft(cid, owner_address="ZION_WALLET_FAKE")
            print("NFT:", nft)
            analytics = await self.get_music_analytics()
            print("Analytics:", analytics)


# Standalone execution demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    music_ai = ZionMusicAI()
    asyncio.run(music_ai.demo())
