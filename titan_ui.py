#!/usr/bin/env python3
"""
TITAN AGENT - Complete Integrated UI
Neural Computer Vision & Action Control
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
import os
import sys
import time
import threading
import queue
import signal
import re
from datetime import datetime
import numpy as np
import random
import math

# Import TITAN backend
from titan_integration import create_backend, VoiceSystem


class FloatingFace(Gtk.Window):
    """Floating face that follows mouse and idles around"""
    
    def __init__(self, parent):
        super().__init__(title="TITAN Face")
        
        self.parent = parent
        
        # Window configuration
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_default_size(150, 150)
        self.set_app_paintable(True)
        
        # Enable transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.set_visual(visual)
        
        # Apply transparent CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background-color: rgba(0, 0, 0, 0);
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            screen,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        # Create face label
        self.face_label = Gtk.Label()
        self.face_label.set_markup('<span font="120">😴</span>')
        
        # Event box for click detection
        event_box = Gtk.EventBox()
        event_box.add(self.face_label)
        event_box.connect("button-press-event", self._on_face_click)
        event_box.set_tooltip_text("Click to pop back in")
        self.add(event_box)
        
        # Movement state
        self.following_mouse = True
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.last_mouse_time = time.time()
        self.mouse_started_moving_time = 0
        self.was_idle = True
        self.idle_offset_x = 0
        self.idle_offset_y = 0
        self.idle_phase = 0
        
        # Velocity damping
        self.velocity_x = 0
        self.velocity_y = 0
        self.max_velocity = 50  # Pixels per frame limit
        
        # Breathing/pulsing effect
        self.pulse_phase = 0
        self.base_font_size = 120
        
        # Start movement loops
        GLib.timeout_add(40, self._update_position)  # Slower: 25 FPS instead of 33
        GLib.timeout_add(100, self._check_mouse_idle)
        GLib.timeout_add(50, self._update_pulse)  # Breathing effect
        
        self.show_all()
    
    def set_face(self, emoji):
        """Update face emoji"""
        current_size = self.base_font_size + int(math.sin(self.pulse_phase) * 5)
        self.face_label.set_markup(f'<span font="{current_size}">{emoji}</span>')
    
    def _update_pulse(self):
        """Create gentle pulsing/breathing effect"""
        self.pulse_phase += 0.2  # Breathing
        
        # Get current emoji
        current_markup = self.face_label.get_label()
        import re
        match = re.search(r'>(.+?)<', current_markup)
        if match:
            emoji = match.group(1)
            # Update with pulsing size
            pulse_size = self.base_font_size + int(math.sin(self.pulse_phase) * 4)
            self.face_label.set_markup(f'<span font="{pulse_size}">{emoji}</span>')
        
        return True
    
    def _on_face_click(self, widget, event):
        """Click to pop back in"""
        if event.type == Gdk.EventType.BUTTON_PRESS and event.button == 1:
            self.parent.pop_in_face()
            return True
        return False
    
    def _check_mouse_idle(self):
        """Check if mouse has been idle"""
        display = Gdk.Display.get_default()
        seat = display.get_default_seat()
        pointer = seat.get_pointer()
        screen, mx, my = pointer.get_position()
        
        # Update last mouse position
        if mx != self.last_mouse_x or my != self.last_mouse_y:
            # Mouse started moving
            if self.was_idle:
                self.mouse_started_moving_time = time.time()
                self.was_idle = False
            
            self.last_mouse_x = mx
            self.last_mouse_y = my
            self.last_mouse_time = time.time()
        else:
            # Mouse is idle
            if not self.was_idle and (time.time() - self.last_mouse_time) > 2.0:
                self.was_idle = True
        
        return True
    
    def _update_position(self):
        """Update face position - follow mouse or idle drift"""
        if not self.get_visible():
            return True
        
        display = Gdk.Display.get_default()
        seat = display.get_default_seat()
        pointer = seat.get_pointer()
        screen, mx, my = pointer.get_position()
        
        # Check if mouse is idle (no movement for 2.5 seconds)
        is_idle = (time.time() - self.last_mouse_time) > 2.5
        
        # Get current position
        current_x, current_y = self.get_position()
        
        if is_idle:
            # Idle animation - gentle bobbing/drifting
            self.idle_phase += 0.04  # Slower drift
            
            # Circular drift pattern with figure-8
            drift_radius = 35
            self.idle_offset_x = math.sin(self.idle_phase) * drift_radius
            self.idle_offset_y = math.cos(self.idle_phase * 0.7) * drift_radius * 0.6
            
            # Add some randomness every so often
            if random.random() < 0.015:  # Less frequent randomness
                self.idle_offset_x += random.uniform(-12, 12)
                self.idle_offset_y += random.uniform(-12, 12)
            
            # Target is last known mouse position + drift
            target_x = self.last_mouse_x + self.idle_offset_x
            target_y = self.last_mouse_y + self.idle_offset_y
            
            # Very smooth idle movement
            smoothing = 0.08
            
        else:
            # Following mode - with startup damping
            time_since_start = time.time() - self.mouse_started_moving_time
            
            # Damping period: first 0.8 seconds after mouse starts moving
            if time_since_start < 2.8:
                # Gradually increase responsiveness (gives time to click)
                damping_factor = time_since_start / 0.8
                smoothing = 0.05 + (0.10 * damping_factor)  # 0.05 → 0.15
                
                # Keep face more stationary during damping
                follow_strength = damping_factor * 0.7
            else:
                # Normal following after damping period
                smoothing = 0.15  # Much smoother than before (was 0.25)
                follow_strength = 1.0
            
            # Follow mouse with offset (not directly on cursor)
            base_offset_x = 80
            base_offset_y = 80
            
            # Add slight variation
            offset_x = base_offset_x + random.randint(-8, 8) * (1 - follow_strength)
            offset_y = base_offset_y + random.randint(-8, 8) * (1 - follow_strength)
            
            # Calculate target with follow strength applied
            target_x = current_x + (mx + offset_x - current_x) * follow_strength
            target_y = current_y + (my + offset_y - current_y) * follow_strength
            
            # Reset idle offsets when following
            self.idle_offset_x *= 0.85
            self.idle_offset_y *= 0.85
        
        # Calculate desired movement
        dx = (target_x - current_x) * smoothing
        dy = (target_y - current_y) * smoothing
        
        # Apply velocity damping (prevents sudden jerks)
        self.velocity_x = self.velocity_x * 0.7 + dx * 0.3
        self.velocity_y = self.velocity_y * 0.7 + dy * 0.3
        
        # Limit maximum velocity (prevents too-fast movement)
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
        
        # Apply movement
        new_x = int(current_x + self.velocity_x)
        new_y = int(current_y + self.velocity_y)
        
        # Move window
        self.move(new_x, new_y)
        
        return True


class TitanUI(Gtk.Window):
    """Complete TITAN Agent Interface"""
    
    def __init__(self):
        super().__init__(title="🧠 TITAN - Neural Agent")
        
        # Window config
        self.set_default_size(500, 600)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_keep_above(True)
        
        # Enable transparency
        self._setup_transparency()
        
        # Transparency state
        self.is_focused = False
        self.is_hovered = False
        self.target_opacity = 0.3  # Default unfocused opacity
        self.current_opacity = 0.3
        
        # Floating face state
        self.face_popped_out = False
        self.floating_face = None
        
        # Initialize backend
        print("🧠 Initializing TITAN Backend...")
        self.backend = create_backend()
        self.voice_system = VoiceSystem()
        
        # Set up callbacks
        self.backend.on_vision_update = self._on_vision_update
        self.backend.on_action_update = self._on_action_update
        
        # UI state
        self.ui_queue = queue.Queue()
        self.last_latent = None
        
        # Build UI
        self._build_ui()
        
        # Connect signals
        self.connect("delete-event", self._on_close)
        self.connect("key-press-event", self._on_key_press)
        self.connect("focus-in-event", self._on_focus_in)
        self.connect("focus-out-event", self._on_focus_out)
        self.connect("enter-notify-event", self._on_mouse_enter)
        self.connect("leave-notify-event", self._on_mouse_leave)
        
        # Enable mouse tracking
        self.add_events(Gdk.EventMask.ENTER_NOTIFY_MASK | Gdk.EventMask.LEAVE_NOTIFY_MASK)
        
        # Start UI update loop
        GLib.timeout_add(100, self._process_ui_queue)
        
        # Start opacity animation loop
        GLib.timeout_add(50, self._animate_opacity)
        
        # Check model status
        self._check_model_status()
        
        print("✓ TITAN UI Ready")
    
    def _setup_transparency(self):
        """Enable window transparency with RGBA visual"""
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        
        if visual and screen.is_composited():
            self.set_visual(visual)
            self.set_app_paintable(True)
            print("  [UI] Transparency enabled")
        else:
            print("  [UI] ⚠ Compositing not available - transparency disabled")
    
    def _animate_opacity(self):
        """Smoothly animate opacity transitions"""
        # Determine target opacity based on focus/hover state
        if self.is_focused or self.is_hovered:
            self.target_opacity = 0.95  # Almost opaque
        else:
            self.target_opacity = 0.35  # Quite transparent
        
        # Smooth transition
        diff = self.target_opacity - self.current_opacity
        if abs(diff) > 0.01:
            self.current_opacity += diff * 0.15  # Smooth interpolation
            self.set_opacity(self.current_opacity)
            
            # Update opacity indicator
            opacity_percent = int(self.current_opacity * 100)
            self.opacity_label.set_markup(f'<span size="small">👁️ {opacity_percent}%</span>')
        
        return True  # Continue animation
    
    def _on_focus_in(self, widget, event):
        """Window gained focus"""
        self.is_focused = True
        return False
    
    def _on_focus_out(self, widget, event):
        """Window lost focus"""
        self.is_focused = False
        return False
    
    def _on_mouse_enter(self, widget, event):
        """Mouse entered window"""
        self.is_hovered = True
        return False
    
    def _on_mouse_leave(self, widget, event):
        """Mouse left window"""
        self.is_hovered = False
        return False
    
    def _on_face_double_click(self, widget, event):
        """Handle double-click on face to pop out/in"""
        if event.type == Gdk.EventType._2BUTTON_PRESS and event.button == 1:
            if self.face_popped_out:
                self.pop_in_face()
            else:
                self.pop_out_face()
            return True
        return False
    
    def pop_out_face(self):
        """Pop the face out into a floating window"""
        if self.face_popped_out:
            return
        
        self.face_popped_out = True
        
        # Hide face in main window
        self.face_event_box.hide()
        
        # Create floating face window
        self.floating_face = FloatingFace(self)
        
        # Sync initial face state
        self._sync_floating_face()
        
        self._log("System", "Face popped out - double-click face to pop back in", "system")
    
    def pop_in_face(self):
        """Pop the face back into the main window"""
        if not self.face_popped_out:
            return
        
        self.face_popped_out = False
        
        # Show face in main window
        self.face_event_box.show()
        
        # Destroy floating window
        if self.floating_face:
            self.floating_face.destroy()
            self.floating_face = None
        
        self._log("System", "Face popped back in", "system")
    
    def _sync_floating_face(self):
        """Sync face emoji to floating window"""
        if self.floating_face and self.face_popped_out:
            # Get current face emoji from main window
            current_markup = self.face_label.get_label()
            # Extract emoji (it's between the span tags)
            import re
            match = re.search(r'>(.+?)<', current_markup)
            if match:
                emoji = match.group(1)
                self.floating_face.set_face(emoji)
    
    def _build_ui(self):
        """Build the interface"""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(main_box)
        
        # Apply styling
        self._apply_theme()
        
        # Compact header with buttons
        header = self._create_compact_header()
        main_box.pack_start(header, False, False, 0)
        
        # Face/Avatar (Igor style)
        face = self._create_face_section()
        main_box.pack_start(face, False, False, 10)
        
        # System Status
        status = self._create_status_section()
        main_box.pack_start(status, False, False, 5)
        
        # Chat/Log
        chat = self._create_chat_section()
        main_box.pack_start(chat, True, True, 5)
        
        # Input
        input_section = self._create_input_section()
        main_box.pack_start(input_section, False, False, 10)
        
        # Status Bar
        status_bar = self._create_status_bar()
        main_box.pack_start(status_bar, False, False, 0)
        
        # Start face animation
        GLib.timeout_add(100, self._animate_face)
        
        self.show_all()
    
    def _apply_theme(self):
        """Apply dark theme CSS with transparency support"""
        css = b"""
            window {
                background-color: rgba(26, 26, 46, 0.95);
            }
            .titan-header {
                background: linear-gradient(90deg, 
                    rgba(22, 33, 62, 0.95) 0%, 
                    rgba(15, 52, 96, 0.95) 100%);
                border-bottom: 2px solid rgba(233, 69, 96, 0.8);
                padding: 5px;
            }
            .control-button {
                background: rgba(15, 52, 96, 0.9);
                color: white;
                border: 1px solid rgba(22, 33, 62, 0.8);
                border-radius: 4px;
                padding: 6px 8px;
                margin: 2px;
                font-size: 11px;
            }
            .control-button:hover {
                background: rgba(22, 33, 62, 0.95);
                border-color: rgba(233, 69, 96, 0.9);
            }
            .active-button {
                background: rgba(233, 69, 96, 0.95);
                border-color: rgba(255, 107, 129, 0.9);
            }
            .chat-view {
                background: rgba(15, 15, 30, 0.85);
                color: #ffffff;
                font-family: monospace;
                font-size: 12px;
            }
            .user-input {
                background: rgba(26, 26, 46, 0.9);
                color: #ffffff;
                border: 1px solid rgba(15, 52, 96, 0.8);
                border-radius: 5px;
                padding: 8px;
            }
        """
        
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
    
    def _create_compact_header(self):
        """Create compact header with title and all control buttons in one row"""
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        vbox.get_style_context().add_class('titan-header')
        
        # Title row
        title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        title_box.set_margin_top(10)
        title_box.set_margin_bottom(5)
        
        title = Gtk.Label()
        title.set_markup('<span font="18" weight="bold">🧠 TITAN</span>')
        title_box.pack_start(title, False, False, 10)
        
        subtitle = Gtk.Label()
        subtitle.set_markup('<span size="small">Neural Agent</span>')
        title_box.pack_start(subtitle, False, False, 0)
        
        vbox.pack_start(title_box, False, False, 0)
        
        # Buttons row - all compact
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        btn_box.set_margin_start(5)
        btn_box.set_margin_end(5)
        btn_box.set_margin_bottom(10)
        btn_box.set_homogeneous(True)
        
        # Start Agent
        self.btn_agent = Gtk.Button(label="▶️ Agent")
        self.btn_agent.get_style_context().add_class('control-button')
        self.btn_agent.set_tooltip_text("Start/Stop autonomous agent")
        self.btn_agent.connect("clicked", self._on_toggle_agent)
        btn_box.pack_start(self.btn_agent, True, True, 0)
        
        # Vision
        self.btn_vision = Gtk.Button(label="👁️ Vision")
        self.btn_vision.get_style_context().add_class('control-button')
        self.btn_vision.set_tooltip_text("Toggle visual processing")
        self.btn_vision.connect("clicked", self._on_toggle_vision)
        btn_box.pack_start(self.btn_vision, True, True, 0)
        
        # Voice
        self.btn_voice = Gtk.Button(label="🎤 Voice")
        self.btn_voice.get_style_context().add_class('control-button')
        self.btn_voice.set_tooltip_text("Toggle voice system")
        self.btn_voice.connect("clicked", self._on_toggle_voice)
        btn_box.pack_start(self.btn_voice, True, True, 0)
        
        # Mute
        self.btn_mute = Gtk.Button(label="🔊")
        self.btn_mute.get_style_context().add_class('control-button')
        self.btn_mute.set_tooltip_text("Toggle sound")
        self.btn_mute.connect("clicked", self._on_toggle_mute)
        btn_box.pack_start(self.btn_mute, True, True, 0)
        
        # Stop
        btn_stop = Gtk.Button(label="⏹️ Stop")
        btn_stop.get_style_context().add_class('control-button')
        btn_stop.set_tooltip_text("Emergency stop all systems")
        btn_stop.connect("clicked", self._on_stop_all)
        btn_box.pack_start(btn_stop, True, True, 0)
        
        # Training
        btn_train = Gtk.Button(label="🏋️")
        btn_train.get_style_context().add_class('control-button')
        btn_train.set_tooltip_text("Open training GUI")
        btn_train.connect("clicked", self._on_open_training)
        btn_box.pack_start(btn_train, True, True, 0)
        
        vbox.pack_start(btn_box, False, False, 0)
        
        return vbox
    
    def _create_face_section(self):
        """Create agent face with Igor-style animations"""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        box.set_margin_start(20)
        box.set_margin_end(20)
        
        # Face - will animate based on state
        self.face_label = Gtk.Label()
        self.face_label.set_markup('<span font="100">😴</span>')
        
        # Event box for double-click detection
        self.face_event_box = Gtk.EventBox()
        self.face_event_box.add(self.face_label)
        self.face_event_box.connect("button-press-event", self._on_face_double_click)
        self.face_event_box.set_tooltip_text("Double-click to pop out/pop in")
        
        box.pack_start(self.face_event_box, False, False, 0)
        
        # Agent name
        self.name_label = Gtk.Label()
        self.name_label.set_markup('<span font="16" weight="bold">TITAN</span>')
        box.pack_start(self.name_label, False, False, 0)
        
        # Status indicator
        self.status_label = Gtk.Label()
        self.status_label.set_markup('<span size="small" color="#808080">● Offline</span>')
        box.pack_start(self.status_label, False, False, 0)
        
        # Animation state
        self.face_state = "offline"
        self.blink_counter = 0
        
        return box
    
    def _animate_face(self):
        """Animate face based on current state (Igor-style)"""
        # Face states and animations
        faces = {
            "offline": "😴",
            "idle": "🙂",
            "thinking": "🤔",
            "active": "🤖",
            "listening": "👂",
            "speaking": "😊",
            "error": "😵"
        }
        
        # Determine current state
        if self.backend.agent_active:
            self.face_state = "active"
        elif self.voice_system.active:
            self.face_state = "listening"
        elif self.backend.vision_active:
            self.face_state = "thinking"
        else:
            self.face_state = "idle" if any([
                self.backend.vae is not None,
                self.backend.policy is not None
            ]) else "offline"
        
        # Blinking effect every 3 seconds
        self.blink_counter += 1
        if self.blink_counter % 30 == 0 and self.face_state in ["idle", "thinking", "active"]:
            # Quick blink
            self.face_label.set_markup('<span font="100">😑</span>')
            GLib.timeout_add(150, lambda: self._set_face_emoji(faces[self.face_state]))
        else:
            # Normal face
            face = faces.get(self.face_state, "🙂")
            self._set_face_emoji(face)
        
        return True  # Continue animation
    
    def _set_face_emoji(self, emoji):
        """Set face emoji and sync to floating window"""
        self.face_label.set_markup(f'<span font="100">{emoji}</span>')
        
        # Sync to floating window if popped out
        if self.face_popped_out and self.floating_face:
            self.floating_face.set_face(emoji)
        
        return False  # Don't repeat this timeout
    
    def _create_status_section(self):
        """Create status indicators"""
        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(5)
        grid.set_margin_start(20)
        grid.set_margin_end(20)
        
        # Vision
        grid.attach(Gtk.Label(label="👁️ Vision:"), 0, 0, 1, 1)
        self.vision_status = Gtk.Label()
        self.vision_status.set_markup('<span size="small" color="#666">Inactive</span>')
        grid.attach(self.vision_status, 1, 0, 1, 1)
        
        # Voice
        grid.attach(Gtk.Label(label="🎤 Voice:"), 0, 1, 1, 1)
        self.voice_status = Gtk.Label()
        self.voice_status.set_markup('<span size="small" color="#666">Inactive</span>')
        grid.attach(self.voice_status, 1, 1, 1, 1)
        
        # Agent
        grid.attach(Gtk.Label(label="🎮 Agent:"), 0, 2, 1, 1)
        self.agent_status = Gtk.Label()
        self.agent_status.set_markup('<span size="small" color="#666">Standby</span>')
        grid.attach(self.agent_status, 1, 2, 1, 1)
        
        return grid
    
    
    def _create_chat_section(self):
        """Create chat/log area"""
        frame = Gtk.Frame(label=" 💬 System Log ")
        frame.set_margin_start(10)
        frame.set_margin_end(10)
        
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(200)
        
        self.chat_view = Gtk.TextView()
        self.chat_view.set_editable(False)
        self.chat_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.chat_view.get_style_context().add_class('chat-view')
        self.chat_view.set_margin_start(10)
        self.chat_view.set_margin_end(10)
        self.chat_view.set_margin_top(10)
        self.chat_view.set_margin_bottom(10)
        
        self.chat_buffer = self.chat_view.get_buffer()
        
        # Tags
        self.chat_buffer.create_tag("system", foreground="#e94560", weight=700)
        self.chat_buffer.create_tag("user", foreground="#4fc3f7", weight=600)
        self.chat_buffer.create_tag("agent", foreground="#66bb6a", weight=600)
        self.chat_buffer.create_tag("timestamp", foreground="#666", size=9000)
        
        scrolled.add(self.chat_view)
        frame.add(scrolled)
        
        self._log("System", "TITAN initialized and ready", "system")
        
        return frame
    
    def _create_input_section(self):
        """Create input area"""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        box.set_margin_start(10)
        box.set_margin_end(10)
        
        self.input_entry = Gtk.Entry()
        self.input_entry.set_placeholder_text("Type command...")
        self.input_entry.get_style_context().add_class('user-input')
        self.input_entry.connect("activate", self._on_send)
        box.pack_start(self.input_entry, True, True, 0)
        
        btn_send = Gtk.Button(label="📤")
        btn_send.get_style_context().add_class('control-button')
        btn_send.connect("clicked", self._on_send)
        box.pack_start(btn_send, False, False, 0)
        
        return box
    
    def _create_status_bar(self):
        """Create status bar"""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.set_margin_start(5)
        box.set_margin_end(5)
        box.set_margin_top(5)
        box.set_margin_bottom(5)
        
        self.fps_label = Gtk.Label()
        self.fps_label.set_markup('<span size="small">FPS: --</span>')
        box.pack_start(self.fps_label, False, False, 0)
        
        # Opacity indicator
        self.opacity_label = Gtk.Label()
        self.opacity_label.set_markup('<span size="small">👁️ 35%</span>')
        box.pack_start(self.opacity_label, False, False, 0)
        
        box.pack_start(Gtk.Label(), True, True, 0)
        
        self.model_label = Gtk.Label()
        self.model_label.set_markup('<span size="small">Models: Loading...</span>')
        box.pack_start(self.model_label, False, False, 0)
        
        return box
    
    # === Event Handlers ===
    
    def _on_toggle_agent(self, btn):
        """Toggle agent"""
        if self.backend.agent_active:
            self.backend.stop_agent()
            btn.set_label("▶️ Agent")
            btn.get_style_context().remove_class('active-button')
            self.agent_status.set_markup('<span size="small" color="#666">Standby</span>')
            self.status_label.set_markup('<span size="small" color="#808080">● Standby</span>')
            self._log("System", "Agent stopped", "system")
        else:
            self.backend.start_agent()
            btn.set_label("⏸️ Agent")
            btn.get_style_context().add_class('active-button')
            self.agent_status.set_markup('<span size="small" color="#66bb6a">● Active</span>')
            self.status_label.set_markup('<span size="small" color="#66bb6a">● Running</span>')
            self._log("System", "Agent started - autonomous control enabled", "system")
    
    def _on_toggle_vision(self, btn):
        """Toggle vision"""
        if self.backend.vision_active:
            self.backend.stop_vision()
            btn.get_style_context().remove_class('active-button')
            self.vision_status.set_markup('<span size="small" color="#666">Inactive</span>')
            self._log("Vision", "Visual processing stopped", "system")
        else:
            self.backend.start_vision()
            btn.get_style_context().add_class('active-button')
            self.vision_status.set_markup('<span size="small" color="#4fc3f7">● Seeing</span>')
            self._log("Vision", "Visual cortex activated", "system")
    
    def _on_toggle_voice(self, btn):
        """Toggle voice"""
        if self.voice_system.active:
            self.voice_system.stop()
            btn.get_style_context().remove_class('active-button')
            self.voice_status.set_markup('<span size="small" color="#666">Inactive</span>')
            self._log("Voice", "Voice input disabled", "system")
        else:
            self.voice_system.start()
            btn.get_style_context().add_class('active-button')
            self.voice_status.set_markup('<span size="small" color="#9c27b0">● Listening</span>')
            self._log("Voice", "Voice verification active", "system")
    
    def _on_toggle_mute(self, btn):
        """Toggle mute"""
        # Placeholder for now
        self._log("System", "Mute toggle (not implemented)", "system")
    
    def _on_stop_all(self, btn):
        """Emergency stop"""
        self.backend.stop_vision()
        self.backend.stop_agent()
        self.voice_system.stop()
        
        # Update UI
        self.btn_agent.set_label("▶️ Agent")
        self.btn_agent.get_style_context().remove_class('active-button')
        self.btn_vision.get_style_context().remove_class('active-button')
        self.btn_voice.get_style_context().remove_class('active-button')
        
        self.agent_status.set_markup('<span size="small" color="#ff5252">● Stopped</span>')
        self.vision_status.set_markup('<span size="small" color="#666">Inactive</span>')
        self.voice_status.set_markup('<span size="small" color="#666">Inactive</span>')
        self.status_label.set_markup('<span size="small" color="#ff5252">● All Systems Halted</span>')
        
        self._log("EMERGENCY", "All systems stopped", "system")
    
    def _on_open_training(self, btn):
        """Open training GUI"""
        import subprocess
        try:
            subprocess.Popen([sys.executable, "titan_gui.py"])
            self._log("System", "Training GUI launched", "system")
        except:
            self._log("System", "Training GUI not found", "system")
    
    def _on_send(self, widget):
        """Send message"""
        text = self.input_entry.get_text().strip()
        if not text:
            return
        
        self.input_entry.set_text("")
        self._log("You", text, "user")
        
        # Simple responses
        if "status" in text.lower():
            info = self.backend.get_model_info()
            msg = f"VAE: {'✓' if info['vae'] else '✗'}, Policy: {'✓' if info['policy'] else '✗'}, Device: {info['device']}"
            self._log("TITAN", msg, "agent")
        elif "hello" in text.lower():
            self._log("TITAN", "Neural systems online. Ready for commands.", "agent")
        else:
            self._log("TITAN", f"Received: {text}", "agent")
    
    def _on_key_press(self, widget, event):
        """Handle keyboard"""
        if event.state & Gdk.ModifierType.CONTROL_MASK and event.keyval == Gdk.KEY_q:
            self._on_close(None, None)
        elif event.keyval == Gdk.KEY_Escape:
            self._on_stop_all(None)
        return False
    
    def _on_close(self, widget, event):
        """Close window"""
        print("\n🧠 Shutting down TITAN...")
        
        # Cleanup floating face if exists
        if self.floating_face:
            self.floating_face.destroy()
        
        self.backend.cleanup()
        Gtk.main_quit()
        return False
    
    # === Callbacks from Backend ===
    
    def _on_vision_update(self, frame, latent, fps):
        """Called when new vision frame is processed"""
        self.ui_queue.put(("vision", (frame, latent, fps)))
    
    def _on_action_update(self, action):
        """Called when action is predicted"""
        # action is numpy array [x, y, click1, click2, scroll, key]
        x, y = action[0], action[1]
        self.ui_queue.put(("action", (x, y)))
    
    # === UI Updates ===
    
    def _process_ui_queue(self):
        """Process updates from backend"""
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()
                
                if msg_type == "vision":
                    frame, latent, fps = data
                    # Just update FPS, no need to display frame
                    self.fps_label.set_markup(f'<span size="small">FPS: {fps:.1f}</span>')
                    self.last_latent = latent
                    
                elif msg_type == "action":
                    x, y = data
                    # Could log action prediction if needed
                    pass
                    
        except queue.Empty:
            pass
        
        return True
    
    def _log(self, sender, message, tag="agent"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        end_iter = self.chat_buffer.get_end_iter()
        
        self.chat_buffer.insert_with_tags_by_name(
            end_iter, f"[{timestamp}] ", "timestamp"
        )
        self.chat_buffer.insert_with_tags_by_name(
            end_iter, f"{sender}: ", tag
        )
        self.chat_buffer.insert(end_iter, f"{message}\n")
        
        # Auto-scroll
        mark = self.chat_buffer.create_mark(None, end_iter, False)
        self.chat_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)
    
    def _check_model_status(self):
        """Check which models are loaded"""
        info = self.backend.get_model_info()
        status = []
        if info['vae']:
            status.append("VAE")
        if info['policy']:
            status.append("Policy")
        if info['cortex']:
            status.append("Cortex")
        
        if status:
            self.model_label.set_markup(f'<span size="small">Models: {", ".join(status)}</span>')
            self._log("System", f"Loaded models: {', '.join(status)}", "system")
        else:
            self.model_label.set_markup('<span size="small" color="#ff5252">No models loaded</span>')
            self._log("System", "⚠ No models found - run training first", "system")


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    print("🧠 Starting TITAN Agent UI...")
    
    app = TitanUI()
    
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\n👋 TITAN terminated")


if __name__ == "__main__":
    main()