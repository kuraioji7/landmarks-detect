import cv2
import mediapipe as mp
# from scipy.spatial import distance
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.5)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 800)
fontScale = 0.5
fontColor = (255, 255, 255)
lineType = 2
topLeftCorner = (20, 20)

custom_landmarks = [
    # horizontal lip points
    61, 291,
    # vertical lip points
    0, 17,
    # topmost and bottommost
    10, 152,
    # leftmost and rightmost
    234, 454,
    #  left eye horizontal, vertical
    33, 133, 159, 145,
    # right eye horizontal, vertical
    362, 263, 386, 374,
    # left eyebrow
    55, 52, 70,
    # right eyebrow
    285, 282, 300,
    # nose
    1,
    # lip inner Center
    13, 14
]

all_points = [i for i in range(468)]

frames_for_average_value = 25
frame_skipper = 25
calibrated = False
variable_dict = dict()

anxiety_array = deque([0] * frames_for_average_value, maxlen=frames_for_average_value)
dist_array = deque([0] * frames_for_average_value, maxlen=frames_for_average_value)
eyebrow_flattning_array = deque([0] * frames_for_average_value, maxlen=frames_for_average_value)
lips_width_array = deque([0] * frames_for_average_value, maxlen=frames_for_average_value)
lips_height_array = deque([0] * frames_for_average_value, maxlen=frames_for_average_value)
eyelid_height_array = deque([0] * frames_for_average_value, maxlen=frames_for_average_value)

emotion_matrix_heu = {
    "lips_height": {"happy": [-10, 37], "sad": [-16, 0], "surprised": [38, 150]},
    "lips_width": {"happy": [10, 48], "sad": [0, 9], "surprised": [-7, -1]},
    "eyebrow_flattning": {"happy": [0, 0], "sad": [-20, -1.5], "surprised": [5, 20]},
    "eyelid_height": {"happy": [0, 0], "sad": [0, 0], "surprised": [10, 30]}
}

min_max = {
    "lips_height": [0, 0],
    "lips_width": [0, 0],
    "eyebrow_flattning": [0, 0],
    "eyelid_height": [0, 0]
}


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def center(a, b):
    return np.mean((a, b), axis=0)


def normalize_and_percent(min_point, max_point, param):
    return ((param - min_point) / (max_point - min_point)) * 100


def average_calculator(data):
    return np.average(np.array(data), axis=0)


def average_value_from_arrays(input_array, input_value):
    input_array.append(input_value)
    return average_calculator(input_array)


def detect_emotion(coords):
    counter = 1
    global calibrated, frame_skipper
    unrefined_lips_width = distance(coords[0], coords[1])
    unrefined_lips_height = distance(coords[2], coords[3])

    eyebrow_center = center(coords[custom_landmarks.index(55)], coords[custom_landmarks.index(70)])
    unrefined_eyebrow_flattning = distance(eyebrow_center, coords[custom_landmarks.index(52)])
    eyebrow_flattning = average_value_from_arrays(eyebrow_flattning_array, unrefined_eyebrow_flattning)

    hor_midpoint = center(coords[0], coords[1])
    ver_midpoint = center(coords[2], coords[3])

    variable_dict["eyebrow_distance"] = distance(coords[custom_landmarks.index(52)],
                                                 coords[custom_landmarks.index(159)])
    variable_dict["eyebrow_flattning"] = average_value_from_arrays(eyebrow_flattning_array, unrefined_eyebrow_flattning)

    unrefined_eyelid_height = distance(coords[custom_landmarks.index(159)], coords[custom_landmarks.index(145)])
    variable_dict["eyelid_height"] = average_value_from_arrays(eyelid_height_array, unrefined_eyelid_height)

    variable_dict["lips_height"] = average_value_from_arrays(lips_height_array, unrefined_lips_height)
    variable_dict["lips_width"] = average_value_from_arrays(lips_width_array, unrefined_lips_width)

    mouth_gap = distance(coords[custom_landmarks.index(13)], coords[custom_landmarks.index(14)])

    percentage_matrix = {
        "lips_height": {"happy": [0, 0.5], "sad": [0, 0.25], "surprised": [0, 1.5]},
        "lips_width": {"happy": [0, 1.5], "sad": [0, 0.25], "surprised": [0, 0.1]},
        "eyebrow_flattning": {"happy": [0, 0], "sad": [0, -2], "surprised": [0, 2]},
        "eyelid_height": {"happy": [0, 0], "sad": [0, 0], "surprised": [0, 1]}
    }

    final_emotion_percentage = {"happy": [0, 0], "sad": [0, 0], "surprised": [0, 0]}
    lips_height_percent = lips_width_percent = eyebrow_flattning_percent = eyelid_height_percent = 0
    #     return mouth_gap

    if calibrated:
        for param in emotion_matrix_heu:
            for emo in emotion_matrix_heu[param]:
                percentage_change = (variable_dict[param] - variable_dict[param + "_threshold"]) * 100 / variable_dict[
                    param + "_threshold"]
                variable_dict[param + "_percent"] = percentage_change
                min_value = emotion_matrix_heu[param][emo][0];
                max_value = emotion_matrix_heu[param][emo][1];

                if min_max[param][0] > percentage_change:
                    min_max[param][0] = percentage_change
                if min_max[param][1] < percentage_change:
                    min_max[param][1] = percentage_change

                if min_value < percentage_change < max_value:
                    temp_percentage = normalize_and_percent(min_value, max_value, percentage_change)
                    bias = percentage_matrix[param][emo][1]

                    if bias != 0:
                        if bias > 0:
                            percentage_matrix[param][emo][0] = temp_percentage
                        else:
                            percentage_matrix[param][emo][0] = 100 - temp_percentage
                        final_emotion_percentage[emo][0] = final_emotion_percentage[emo][0] + \
                                                           percentage_matrix[param][emo][0] * abs(bias)
                        final_emotion_percentage[emo][1] += 1
            counter += 1

        for emo in final_emotion_percentage:
            if final_emotion_percentage[emo][1] == 0:
                final_emotion_percentage[emo][0] = 0
            else:
                final_emotion_percentage[emo][0] = (final_emotion_percentage[emo][0] / final_emotion_percentage[emo][1])

        return final_emotion_percentage

    else:
        if frame_skipper == 0:
            variable_dict["anxiety_threshold"] = variable_dict["eyebrow_distance"]
            variable_dict["eyebrow_flattning_threshold"] = unrefined_eyebrow_flattning
            variable_dict["lips_height_threshold"] = variable_dict["lips_height"]
            variable_dict["lips_width_threshold"] = variable_dict["lips_width"]
            variable_dict["eyelid_height_threshold"] = variable_dict["eyelid_height"]
            calibrated = True
        else:
            frame_skipper -= 1


def process_and_detect(input_frame):
    height, width, channels = input_frame.shape
    coords = []

    results = face_mesh.process(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmark_list = list(results.multi_face_landmarks[0].landmark)

        temp = input_frame.copy()
        coords = []
        for i in custom_landmarks:
            #             coords.append([landmark_list[i].x, landmark_list[i].y])
            coord_x = int(landmark_list[i].x * width)
            coord_y = int(landmark_list[i].y * height)
            coords.append([coord_x, coord_y])

            temp = cv2.circle(temp, (coord_x, coord_y), radius=2, color=(0, 255, 0), thickness=1)

        emotion = detect_emotion(coords)
        if emotion:
            top_left = [20, 20]
            temp = cv2.flip(temp, 1)
            for key, value in emotion.items():
                output_string = key + ":" + str(value[0])
                cv2.putText(temp, output_string, tuple(top_left), font, fontScale, fontColor, lineType)
                top_left[1] += 30
            # cv2.imshow("test", temp)
            return temp
        else:
            return None
    else:
        # cv2.imshow("test", cv2.flip(input_frame, 1))
        return cv2.flip(input_frame, 1)
