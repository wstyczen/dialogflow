# Dialogflow

- [Dialogflow](#dialogflow)
  - [Dependencies](#dependencies)
    - [Python libraries](#python-libraries)
    - [ROS packages](#ros-packages)
      - [Additional packages necessary for running the extended VAD](#additional-packages-necessary-for-running-the-extended-vad)
  - [VAD](#vad)
    - [Functionality](#functionality)
    - [Running VAD](#running-vad)
  - [Voice communication system](#voice-communication-system)
    - [Functionality](#functionality-1)
      - [Basic working scenario](#basic-working-scenario)
      - [Fallback actions](#fallback-actions)
    - [Running the voice communication system](#running-the-voice-communication-system)

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

## Voice communication system

### Functionality

> Utilizes the [VAD](#vad) functionality to listen for voice commands and extends it with additional steps and fallback actions.

#### Basic working scenario

1. Run wake-word detection.
2. Upon detecting the wake-word orient the robot towards the human (can take some time, depending on the movement required).
3. Robot asks the person to give their voice command.
4. Recording of the command starts.
5. When the recording finishes, the audio is saved to a .wav file and the path to the audio is published.
6. Audio is passed to [the sound processing package](https://github.com/wstyczen/sound_processing) and its quality is enhanced.
7. Speech-to-text is performed on the recorded audio.
8. The text interpretation of the voice command is passed to intent detection for further processing (UNIMPLEMENTED).

#### Fallback actions

> Various fallback actions are performed in case of failure of any of the above steps.
>
> Implementation in progress.
> **TODO: Describe the fallback actions and their execution conditions when done.**

### Running the voice communication system

```sh
# Run the RVIZ Rico simulation.
roslaunch human_interactions run_simulation.launch
# Run the human interactions action servers.
roslaunch human_interactions action_servers.launch
# Run the extended vad node.
roslaunch dialogflow voice_communication.launch
```
