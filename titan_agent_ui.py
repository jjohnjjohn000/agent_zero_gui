#!/usr/bin/env python3
"""
TITAN AGENT UI - Neural Computer Control Interface
Inspired by Igor's design, adapted for vision-action-voice system
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
from datetime import datetime

# Add models directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

class TitanAgentWindow(Gtk.Window):
    """Main TITAN Agent Window - Vision-Action-Voice Interface"""
    
    def __init__(self):
        super().__init__(title="🧠 TITAN - Neural Agent")
        
        # Window configuration
        self.set_default_size(500, 700)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_keep_above(True)  # Always on top like Igor
        
        # System state
        self.agent_active = False
        self.voice_active = False
        self.vision_active = False
        self.is_muted = False
        self.is_listening = False
        
        # Queues for thread-safe communication
        self.ui_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Build the interface
        self._build_ui()
        
        # Connect signals
        self.connect("delete-event", self._on_close)
        self.connect("key-press-event", self._on_key_press)
        
        # Start UI update loop
        GLib.timeout_add(100, self._process_ui_queue)
        
        print("🧠 TITAN Agent UI initialized")
    
    def _build_ui(self):
        """Build the complete user interface"""
        
        # Main container with dark theme
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(main_box)
        
        # Apply dark theme
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1e 100%);
            }
            .titan-header {
                background: linear-gradient(90deg, #16213e 0%, #0f3460 100%);
                border-bottom: 2px solid #e94560;
                padding: 15px;
            }
            .titan-title {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
            }
            .titan-subtitle {
                color: #a0a0a0;
                font-size: 11px;
            }
            .status-bar {
                background: #1a1a2e;
                border-top: 1px solid #333;
                padding: 8px;
            }
            .control-button {
                background: #0f3460;
                color: white;
                border: 1px solid #16213e;
                border-radius: 5px;
                padding: 10px;
                margin: 3px;
            }
            .control-button:hover {
                background: #16213e;
                border-color: #e94560;
            }
            .active-button {
                background: #e94560;
                border-color: #ff6b81;
            }
            .chat-view {
                background: #0f0f1e;
                color: #ffffff;
                font-family: monospace;
                font-size: 12px;
            }
            .user-input {
                background: #1a1a2e;
                color: #ffffff;
                border: 1px solid #0f3460;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        
        style_context = self.get_style_context()
        style_context.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        # === HEADER SECTION ===
        header = self._create_header()
        main_box.pack_start(header, False, False, 0)
        
        # === FACE/AVATAR SECTION ===
        face_frame = self._create_face_section()
        main_box.pack_start(face_frame, False, False, 10)
        
        # === SYSTEM STATUS SECTION ===
        status_section = self._create_status_section()
        main_box.pack_start(status_section, False, False, 5)
        
        # === CONTROL BUTTONS ===
        controls = self._create_control_buttons()
        main_box.pack_start(controls, False, False, 10)
        
        # === CHAT/LOG SECTION ===
        chat_frame = self._create_chat_section()
        main_box.pack_start(chat_frame, True, True, 5)
        
        # === INPUT SECTION ===
        input_section = self._create_input_section()
        main_box.pack_start(input_section, False, False, 10)
        
        # === STATUS BAR ===
        status_bar = self._create_status_bar()
        main_box.pack_start(status_bar, False, False, 0)
        
        self.show_all()
    
    def _create_header(self):
        """Create the header with title and subtitle"""
        header = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        header.get_style_context().add_class('titan-header')
        
        # Title
        title_label = Gtk.Label()
        title_label.set_markup('<span font="24" weight="bold">🧠 TITAN</span>')
        title_label.get_style_context().add_class('titan-title')
        header.pack_start(title_label, False, False, 0)
        
        # Subtitle
        subtitle = Gtk.Label()
        subtitle.set_markup('<span size="small">Neural Computer Vision & Action System</span>')
        subtitle.get_style_context().add_class('titan-subtitle')
        header.pack_start(subtitle, False, False, 0)
        
        return header
    
    def _create_face_section(self):
        """Create the agent face/avatar section (Igor-style)"""
        frame = Gtk.Frame()
        frame.set_shadow_type(Gtk.ShadowType.NONE)
        
        face_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        face_box.set_margin_start(20)
        face_box.set_margin_end(20)
        
        # Avatar/Face - using emoji for now (can be replaced with actual image)
        self.face_label = Gtk.Label()
        self.face_label.set_markup('<span font="80">🤖</span>')
        face_box.pack_start(self.face_label, False, False, 0)
        
        # Agent name
        self.name_label = Gtk.Label()
        self.name_label.set_markup('<span font="16" weight="bold">TITAN Agent</span>')
        face_box.pack_start(self.name_label, False, False, 0)
        
        # Status indicator
        self.status_label = Gtk.Label()
        self.status_label.set_markup('<span size="small" color="#808080">● Offline</span>')
        face_box.pack_start(self.status_label, False, False, 0)
        
        frame.add(face_box)
        return frame
    
    def _create_status_section(self):
        """Create system status indicators"""
        grid = Gtk.Grid()
        grid.set_column_spacing(10)
        grid.set_row_spacing(5)
        grid.set_margin_start(20)
        grid.set_margin_end(20)
        
        # Vision status
        vision_label = Gtk.Label()
        vision_label.set_markup('<span size="small">👁️ Vision:</span>')
        self.vision_status = Gtk.Label()
        self.vision_status.set_markup('<span size="small" color="#666666">Inactive</span>')
        grid.attach(vision_label, 0, 0, 1, 1)
        grid.attach(self.vision_status, 1, 0, 1, 1)
        
        # Voice status
        voice_label = Gtk.Label()
        voice_label.set_markup('<span size="small">🎤 Voice:</span>')
        self.voice_status = Gtk.Label()
        self.voice_status.set_markup('<span size="small" color="#666666">Inactive</span>')
        grid.attach(voice_label, 0, 1, 1, 1)
        grid.attach(self.voice_status, 1, 1, 1, 1)
        
        # Agent status
        agent_label = Gtk.Label()
        agent_label.set_markup('<span size="small">🎮 Agent:</span>')
        self.agent_status = Gtk.Label()
        self.agent_status.set_markup('<span size="small" color="#666666">Standby</span>')
        grid.attach(agent_label, 0, 2, 1, 1)
        grid.attach(self.agent_status, 1, 2, 1, 1)
        
        return grid
    
    def _create_control_buttons(self):
        """Create control buttons grid"""
        grid = Gtk.Grid()
        grid.set_column_spacing(5)
        grid.set_row_spacing(5)
        grid.set_margin_start(20)
        grid.set_margin_end(20)
        grid.set_column_homogeneous(True)
        
        # Row 1: Main controls
        self.btn_agent = Gtk.Button(label="▶️ START AGENT")
        self.btn_agent.get_style_context().add_class('control-button')
        self.btn_agent.connect("clicked", self._on_toggle_agent)
        grid.attach(self.btn_agent, 0, 0, 1, 1)
        
        self.btn_vision = Gtk.Button(label="👁️ VISION")
        self.btn_vision.get_style_context().add_class('control-button')
        self.btn_vision.connect("clicked", self._on_toggle_vision)
        grid.attach(self.btn_vision, 1, 0, 1, 1)
        
        # Row 2: Voice & Mute
        self.btn_voice = Gtk.Button(label="🎤 VOICE")
        self.btn_voice.get_style_context().add_class('control-button')
        self.btn_voice.connect("clicked", self._on_toggle_voice)
        grid.attach(self.btn_voice, 0, 1, 1, 1)
        
        self.btn_mute = Gtk.Button(label="🔊 SOUND ON")
        self.btn_mute.get_style_context().add_class('control-button')
        self.btn_mute.connect("clicked", self._on_toggle_mute)
        grid.attach(self.btn_mute, 1, 1, 1, 1)
        
        # Row 3: Utility buttons
        btn_stop = Gtk.Button(label="⏹️ STOP ALL")
        btn_stop.get_style_context().add_class('control-button')
        btn_stop.connect("clicked", self._on_stop_all)
        grid.attach(btn_stop, 0, 2, 1, 1)
        
        btn_train = Gtk.Button(label="🏋️ TRAINING")
        btn_train.get_style_context().add_class('control-button')
        btn_train.connect("clicked", self._on_open_training)
        grid.attach(btn_train, 1, 2, 1, 1)
        
        return grid
    
    def _create_chat_section(self):
        """Create the chat/log display"""
        frame = Gtk.Frame(label=" 💬 System Log ")
        frame.set_margin_start(10)
        frame.set_margin_end(10)
        
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(250)
        
        self.chat_view = Gtk.TextView()
        self.chat_view.set_editable(False)
        self.chat_view.set_cursor_visible(False)
        self.chat_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.chat_view.get_style_context().add_class('chat-view')
        self.chat_view.set_margin_start(10)
        self.chat_view.set_margin_end(10)
        self.chat_view.set_margin_top(10)
        self.chat_view.set_margin_bottom(10)
        
        self.chat_buffer = self.chat_view.get_buffer()
        
        # Create text tags for formatting
        self.chat_buffer.create_tag("system", foreground="#e94560", weight=700)
        self.chat_buffer.create_tag("user", foreground="#4fc3f7", weight=600)
        self.chat_buffer.create_tag("agent", foreground="#66bb6a", weight=600)
        self.chat_buffer.create_tag("timestamp", foreground="#666666", size=9000)
        self.chat_buffer.create_tag("error", foreground="#ff5252", weight=600)
        
        scrolled.add(self.chat_view)
        frame.add(scrolled)
        
        # Add welcome message
        self._add_log_message("System", "TITAN Agent UI initialized. Ready for neural operations.", "system")
        
        return frame
    
    def _create_input_section(self):
        """Create the user input section"""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        box.set_margin_start(10)
        box.set_margin_end(10)
        
        self.input_entry = Gtk.Entry()
        self.input_entry.set_placeholder_text("Type command or question...")
        self.input_entry.get_style_context().add_class('user-input')
        self.input_entry.connect("activate", self._on_send_message)
        box.pack_start(self.input_entry, True, True, 0)
        
        btn_send = Gtk.Button(label="📤 Send")
        btn_send.get_style_context().add_class('control-button')
        btn_send.connect("clicked", self._on_send_message)
        box.pack_start(btn_send, False, False, 0)
        
        return box
    
    def _create_status_bar(self):
        """Create bottom status bar"""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.get_style_context().add_class('status-bar')
        
        # FPS/Performance indicator
        self.fps_label = Gtk.Label()
        self.fps_label.set_markup('<span size="small">FPS: --</span>')
        box.pack_start(self.fps_label, False, False, 5)
        
        # Spacer
        box.pack_start(Gtk.Label(), True, True, 0)
        
        # Model info
        self.model_label = Gtk.Label()
        self.model_label.set_markup('<span size="small">VAE: 4096-dim | Policy: Ready</span>')
        box.pack_start(self.model_label, False, False, 5)
        
        return box
    
    # === EVENT HANDLERS ===
    
    def _on_toggle_agent(self, button):
        """Toggle the autonomous agent"""
        self.agent_active = not self.agent_active
        
        if self.agent_active:
            button.set_label("⏸️ PAUSE AGENT")
            button.get_style_context().add_class('active-button')
            self.agent_status.set_markup('<span size="small" color="#66bb6a">● Active</span>')
            self.status_label.set_markup('<span size="small" color="#66bb6a">● Agent Running</span>')
            self.face_label.set_markup('<span font="80">🤖</span>')
            self._add_log_message("System", "Agent started - autonomous control enabled", "system")
            
            # Start agent in background thread
            threading.Thread(target=self._agent_loop, daemon=True).start()
        else:
            button.set_label("▶️ START AGENT")
            button.get_style_context().remove_class('active-button')
            self.agent_status.set_markup('<span size="small" color="#808080">● Standby</span>')
            self.status_label.set_markup('<span size="small" color="#808080">● Standby</span>')
            self._add_log_message("System", "Agent paused", "system")
    
    def _on_toggle_vision(self, button):
        """Toggle vision system"""
        self.vision_active = not self.vision_active
        
        if self.vision_active:
            button.get_style_context().add_class('active-button')
            self.vision_status.set_markup('<span size="small" color="#4fc3f7">● Seeing</span>')
            self._add_log_message("Vision", "Visual cortex activated - encoding frames", "system")
        else:
            button.get_style_context().remove_class('active-button')
            self.vision_status.set_markup('<span size="small" color="#666666">Inactive</span>')
            self._add_log_message("Vision", "Visual input suspended", "system")
    
    def _on_toggle_voice(self, button):
        """Toggle voice system"""
        self.voice_active = not self.voice_active
        
        if self.voice_active:
            button.get_style_context().add_class('active-button')
            self.voice_status.set_markup('<span size="small" color="#9c27b0">● Listening</span>')
            self._add_log_message("Voice", "Speaker verification active", "system")
        else:
            button.get_style_context().remove_class('active-button')
            self.voice_status.set_markup('<span size="small" color="#666666">Inactive</span>')
            self._add_log_message("Voice", "Voice input disabled", "system")
    
    def _on_toggle_mute(self, button):
        """Toggle mute"""
        self.is_muted = not self.is_muted
        
        if self.is_muted:
            button.set_label("🔇 MUTED")
            self._add_log_message("System", "Audio output muted", "system")
        else:
            button.set_label("🔊 SOUND ON")
            self._add_log_message("System", "Audio output enabled", "system")
    
    def _on_stop_all(self, button):
        """Emergency stop - halt all systems"""
        self.agent_active = False
        self.vision_active = False
        self.voice_active = False
        
        # Update UI
        self.btn_agent.set_label("▶️ START AGENT")
        self.btn_agent.get_style_context().remove_class('active-button')
        self.btn_vision.get_style_context().remove_class('active-button')
        self.btn_voice.get_style_context().remove_class('active-button')
        
        self.agent_status.set_markup('<span size="small" color="#ff5252">● Emergency Stop</span>')
        self.vision_status.set_markup('<span size="small" color="#666666">Inactive</span>')
        self.voice_status.set_markup('<span size="small" color="#666666">Inactive</span>')
        self.status_label.set_markup('<span size="small" color="#ff5252">● All Systems Halted</span>')
        
        self._add_log_message("EMERGENCY", "All systems stopped by user", "error")
    
    def _on_open_training(self, button):
        """Open the training GUI"""
        self._add_log_message("System", "Opening training interface...", "system")
        
        # Launch titan_gui in separate process
        import subprocess
        try:
            subprocess.Popen([sys.executable, "titan_gui.py"])
            self._add_log_message("System", "Training GUI launched", "system")
        except Exception as e:
            self._add_log_message("Error", f"Failed to launch training: {e}", "error")
    
    def _on_send_message(self, widget):
        """Handle user text input"""
        text = self.input_entry.get_text().strip()
        if not text:
            return
        
        self.input_entry.set_text("")
        self._add_log_message("You", text, "user")
        
        # Process command in background
        threading.Thread(target=self._process_command, args=(text,), daemon=True).start()
    
    def _on_key_press(self, widget, event):
        """Handle keyboard shortcuts"""
        # Ctrl+Q to quit
        if event.state & Gdk.ModifierType.CONTROL_MASK and event.keyval == Gdk.KEY_q:
            self._on_close(None, None)
            return True
        
        # ESC to stop agent
        if event.keyval == Gdk.KEY_Escape:
            if self.agent_active:
                self._on_stop_all(None)
            return True
        
        return False
    
    def _on_close(self, widget, event):
        """Handle window close"""
        print("\n🧠 Shutting down TITAN Agent...")
        self.agent_active = False
        self.vision_active = False
        self.voice_active = False
        
        # Give threads time to clean up
        time.sleep(0.5)
        
        Gtk.main_quit()
        return False
    
    # === CORE LOGIC ===
    
    def _agent_loop(self):
        """Main agent control loop (runs in background thread)"""
        print("🤖 Agent control loop started")
        frame_count = 0
        
        try:
            while self.agent_active:
                frame_count += 1
                
                # Simulate vision processing
                if self.vision_active and frame_count % 10 == 0:
                    self.ui_queue.put(("fps", f"FPS: {frame_count % 60}"))
                
                # Simulate agent thinking
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Agent loop error: {e}")
            self.ui_queue.put(("log", ("Error", f"Agent error: {e}", "error")))
    
    def _process_command(self, text):
        """Process user command (runs in background thread)"""
        try:
            # Simple command processing
            text_lower = text.lower()
            
            if "hello" in text_lower or "hi" in text_lower:
                response = "Hello! TITAN neural agent ready for commands."
            elif "status" in text_lower:
                response = f"Agent: {'Active' if self.agent_active else 'Standby'}, Vision: {'On' if self.vision_active else 'Off'}, Voice: {'On' if self.voice_active else 'Off'}"
            elif "help" in text_lower:
                response = "Commands: Use buttons to control systems, or type questions here. ESC = Emergency stop, Ctrl+Q = Quit"
            else:
                response = f"Command received: {text}"
            
            self.ui_queue.put(("log", ("TITAN", response, "agent")))
            
        except Exception as e:
            self.ui_queue.put(("log", ("Error", str(e), "error")))
    
    def _add_log_message(self, sender, message, tag="agent"):
        """Add message to chat log (thread-safe via queue)"""
        self.ui_queue.put(("log", (sender, message, tag)))
    
    def _process_ui_queue(self):
        """Process UI updates from background threads (runs in main GTK thread)"""
        try:
            while True:
                msg_type, data = self.ui_queue.get_nowait()
                
                if msg_type == "log":
                    sender, message, tag = data
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    end_iter = self.chat_buffer.get_end_iter()
                    
                    # Add timestamp
                    self.chat_buffer.insert_with_tags_by_name(
                        end_iter, f"[{timestamp}] ", "timestamp"
                    )
                    
                    # Add sender
                    self.chat_buffer.insert_with_tags_by_name(
                        end_iter, f"{sender}: ", tag
                    )
                    
                    # Add message
                    self.chat_buffer.insert(end_iter, f"{message}\n")
                    
                    # Auto-scroll to bottom
                    mark = self.chat_buffer.create_mark(None, end_iter, False)
                    self.chat_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)
                
                elif msg_type == "fps":
                    self.fps_label.set_markup(f'<span size="small">{data}</span>')
                    
        except queue.Empty:
            pass
        
        return True  # Continue calling this function


def main():
    """Main entry point"""
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    print("🧠 Starting TITAN Agent UI...")
    win = TitanAgentWindow()
    
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\n👋 TITAN Agent terminated by user")


if __name__ == "__main__":
    main()
