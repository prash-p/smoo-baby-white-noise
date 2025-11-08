from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
from pathlib import Path

# Import your existing code
import sys
sys.path.append('.')
from main import AudioController, CONFIG

app = Flask(__name__)
CORS(app)

# Global controller instance
controller = None
controller_thread = None
is_running = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of the audio system."""
    global controller, is_running
    
    if controller and is_running:
        return jsonify({
            'running': True,
            'current_level': controller.current_level,
            'max_level': 4,
            'volume_percent': CONFIG['volume_levels'][controller.current_level],
            'power_threshold': CONFIG['power_threshold'],
            'level_down_time': CONFIG['level_down_time'],
            'level_up_time': CONFIG['level_up_time']
        })
    else:
        return jsonify({
            'running': False,
            'current_level': CONFIG['current_level'],
            'max_level': 4,
            'volume_percent': CONFIG['volume_levels'][CONFIG['current_level']],
            'power_threshold': CONFIG['power_threshold'],
            'level_down_time': CONFIG['level_down_time'],
            'level_up_time': CONFIG['level_up_time']
        })

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the audio monitoring system."""
    global controller, controller_thread, is_running
    
    if is_running:
        return jsonify({'success': False, 'message': 'System already running'})
    
    try:
        # Create controller only if it doesn't exist
        if controller is None:
            controller = AudioController(config=CONFIG)
        
        is_running = True
        CONFIG['current_level'] = controller.current_level = 0
        controller.player = controller.preloaded_players[0]
        controller.player.start()
        controller.monitor.start()
        
        return jsonify({'success': True, 'message': 'System started successfully'})
    except Exception as e:
        is_running = False
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the audio monitoring system."""
    global controller, is_running
    
    if not is_running:
        return jsonify({'success': False, 'message': 'System not running'})
    
    try:
        is_running = False
        if controller:
            # Stop player and monitor, but don't destroy the controller
            controller.player.stop()
            controller.monitor.stop()
        
        return jsonify({'success': True, 'message': 'System stopped successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/level', methods=['POST'])
def set_level():
    """Manually set the sound level."""
    global controller
    
    if not is_running or not controller:
        return jsonify({'success': False, 'message': 'System not running'})
    
    data = request.get_json()
    new_level = data.get('level')
    
    if new_level is None or new_level < 0 or new_level > 4:
        return jsonify({'success': False, 'message': 'Invalid level (must be 0-4)'})
    
    try:
        controller._handle_level_change(int(new_level))
        return jsonify({'success': True, 'message': f'Level set to {new_level}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration parameters."""
    global controller
    
    data = request.get_json()
    
    if 'level_down_time' in data:
        CONFIG['level_down_time'] = int(data['level_down_time'])
    
    if 'level_up_time' in data:
        CONFIG['level_up_time'] = int(data['level_up_time'])
    
    return jsonify({'success': True, 'message': 'Configuration updated'})

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Fully shutdown and cleanup resources."""
    global controller, is_running
    
    try:
        is_running = False
        if controller:
            controller.cleanup()  # Use cleanup instead of stop
            controller = None
        
        return jsonify({'success': True, 'message': 'System shutdown complete'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    import atexit
    
    # Register cleanup on exit
    def cleanup_on_exit():
        global controller, is_running
        if controller:
            is_running = False
            controller.cleanup()
    
    atexit.register(cleanup_on_exit)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
