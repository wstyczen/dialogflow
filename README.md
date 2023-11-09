# Dialogflow

- [Dialogflow](#dialogflow)
  - [Dependencies](#dependencies)
    - [Python libraries](#python-libraries)
    - [ROS packages](#ros-packages)
      - [Additional packages necessary for running the extended VAD](#additional-packages-necessary-for-running-the-extended-vad)
  - [VAD](#vad)
    - [Functionality](#functionality)
    - [Running VAD](#running-vad)
  - [Extended VAD](#extended-vad)
    - [Functionality](#functionality-1)
      - [Basic extended scenario](#basic-extended-scenario)
      - [Emergency action](#emergency-action)
    - [Running extended VAD](#running-extended-vad)

## Dependencies

### Python libraries

> Python 3.6 or greater.

```sh
pip3 install gtts numpy pyaudio pygame pvporcupine scipy SpeechRecognition
```

### ROS packages

> [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) full installation.

#### Additional packages necessary for running [the extended VAD](#extended-vad)

- [Human Interactions](https://github.com/wstyczen/human_interactions)
- [Sound Processing](https://github.com/wstyczen/sound_processing)

## VAD

### Functionality

> _Voice Activation Detector_ - implements _wake-word_ detection by streaming audio from the microphone(s) to the [porcupine engines](https://github.com/Picovoice/porcupine).
>
> Upon detection allows for recording voice commands by keeping count of voiced frames in real time to judge whether a sentence was recorded. Optionally also guarded by a time limit.
>
> Wake word detection can be simulated manually to start recording by posting to a topic from command line:
>
> `rostopic pub /start_recording std_msgs/Bool 1`.
>
> When the voice command finishes recording it is saved to a .wav file and its path is published / returned.

### Running VAD

```sh
roslaunch dialogflow vad.launch
```

## Extended VAD

### Functionality

> Extends the [VAD](#vad) with features like robot movement in relation to the person's location, sound enhancement or voice comunication.

#### Basic extended scenario

1. Run wake-word detection.
2. Upon detecting the wake-word orient the robot towards the human (can take some time, depending on the movement required).
3. Robot asks the person to give their voice command.
4. Recording of the command starts.
5. When the recording finishes, the audio is saved to a .wav file and the path to the audio is returned.
6. Audio is passed to [the sound processing package](https://github.com/wstyczen/sound_processing) and its quality is (hopefully) enhanced.
7. Speech-to-text is performed on the recorded audio. If it can't be inferred with high enough probability the [emergency_action](#emergency-action) is performed.
8. The text intepreted from voice command is passed to intent detection for whatever use.

#### Emergency action

1. Voice notification about performing the action.
2. Moving nearby the person for better audio quality.
3. Asking the person to repeat the voice command.
4. Start to record again.

### Running extended VAD

```sh
# Run the RVIZ Rico simulation.
roslaunch human_interactions run_simulation.launch
# Run the human interactions action servers.
roslaunch human_interactions action_servers.launch
# Run the extended vad node.
roslaunch dialogflow extended_vad.launch
```
