import mido  # for MIDI file handling
import pygame  # for animation
import random

# Global variables for screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


def parse_midi(file_path):
    midi = mido.MidiFile(file_path)
    ticks_per_beat = midi.ticks_per_beat
    microseconds_per_beat = 500000  # Default tempo is 120 BPM, which is 500000 microseconds per beat
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                microseconds_per_beat = msg.tempo
                break

    # Calculate milliseconds per tick for later use
    ms_per_tick = microseconds_per_beat / ticks_per_beat / 1000

    # Now, process the events
    events = []
    elapsed_time = 0  # This will keep track of the cumulative time in ticks
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta:
                elapsed_time += msg.time
                if msg.type == 'note_on' or msg.type == 'note_off':
                    event = {
                        'type': msg.type,
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'time': elapsed_time * ms_per_tick  # Convert time to milliseconds
                    }
                    events.append(event)

    return events, ms_per_tick


def initialize_animation():
    pygame.init()  # Initialize all imported Pygame modules

    # Set up the drawing window
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

    # Set the title of the window
    pygame.display.set_caption('MIDI Animation')

    return screen


def update_animation(screen, midi_events, current_time, active_notes):

    # Remove notes that have finished playing
    active_notes = [note for note in active_notes if current_time < note['end_time']]

    # Process new MIDI events
    while midi_events and midi_events[0]['time'] <= current_time:
        # print(midi_events[0]['time'], current_time)
        event = midi_events.pop(0)
        # print(event['type'])
        if event['type'] == 'note_on':
            # Calculate the position and size for the circle
            x = (event['note'] % (SCREEN_WIDTH // 2)) * 10
            y = SCREEN_HEIGHT // 2
            size = max(10, event['velocity'] // 2) * 3 - 25
            print(size)

            # Find corresponding note_off event to get the duration
            end_time = current_time
            for off_event in midi_events:
                if off_event['type'] == 'note_off' and off_event['note'] == event['note']:
                    end_time += off_event['time']
                    break

            # Add the note to the active notes
            active_notes.append({
                'x': x,
                'y': y,
                'size': size,
                'end_time': end_time
            })

    # Draw active notes
    for note in active_notes:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pygame.draw.circle(screen, random_color, (note['x'], note['y']), note['size'])

    # Update the display
    pygame.display.flip()


def get_current_time():
    # Return the current time in milliseconds
    return pygame.time.get_ticks()


def main():
    midi_events, ms_per_tick = parse_midi('midi-file/MIDI_sample.mid')
    screen = initialize_animation()
    active_notes = []

    start_time = pygame.time.get_ticks()

    running = True
    while running:
        # Calculate the current time in milliseconds since the start of the animation
        current_time = pygame.time.get_ticks() - start_time

        # Update and render the animation based on the current time
        update_animation(screen, midi_events, current_time, active_notes)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


main()
