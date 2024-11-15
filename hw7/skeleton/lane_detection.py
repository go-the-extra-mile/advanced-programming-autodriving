import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import cv2


class LaneDetection:
    """
    Lane detection module using edge detection and b-spline fitting

    args:
        cut_size (cut_size=68) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    """

    def __init__(
        self,
        cut_size=65,
        spline_smoothness=16,
        gradient_threshold=11,
        distance_maxima_gradient=5,
    ):
        self.car_position = np.array([48, 0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0

    def cut_gray(self, state_image_full):
        """
        This function should cut the image at the front end of the car (e.g. pixel row 68)
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 68x96x1

        """
        # crop image to keep only the top cut_size pixels
        state_image_cut = state_image_full[: self.cut_size, :, :]

        # convert image to grayscale
        gray_state_image = cv2.cvtColor(state_image_cut, cv2.COLOR_BGR2GRAY)

        # expand dimension to get a shape of (cut_size, 96, 1)
        gray_state_image = np.expand_dims(gray_state_image, axis=2)

        # return image that is flipped upside down
        return gray_state_image[::-1]

    def edge_detection(self, gray_image):
        """
        In order to find edges in the gray state image,
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel.
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero.

        input:
            gray_state_image 68x96x1

        output:
            gradient_sum 68x96x1

        """
        gradients = np.gradient(gray_image, axis=(0, 1))
        # gradients[0]: column-wise gradient
        # gradients[1]: row-wise gradient

        abs_gradients = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
        abs_gradients[abs_gradients < self.gradient_threshold] = 0
        abs_gradients = np.expand_dims(abs_gradients, axis=2)

        return abs_gradients

    def find_maxima_gradient_rowwise(self, gradient_sum):
        """
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima.
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 68x96x1

        output:
            maxima (np.array) 2x Number_maxima

        """
        maxima_indices = []
        max_len = 0

        gradient_sum_mat = np.squeeze(gradient_sum)
        for row in gradient_sum_mat:
            # Find the indices of the local maxima in the row
            peaks, _ = find_peaks(row, distance=self.distance_maxima_gradient)

            max_len = max(max_len, len(peaks))

            # Append the indices to the list
            maxima_indices.append(peaks)

        maxima_indices = [
            np.pad(i, (0, max_len - len(i)), "constant", constant_values=0)
            for i in maxima_indices
        ]

        # Convert the list of lists to a numpy ndarray and return
        return np.array(maxima_indices)

    def find_first_lane_point(self, gradient_sum):
        """
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered.
        Even though there is space for improvement ;)

        input:
            gradient_sum 68x96x1

        output:
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        """

        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # MODIFICATION
        gradient_sum = np.squeeze(gradient_sum)

        # loop through the rows
        while not lanes_found:
            # Find peaks with min distance of at least 3 pixel
            argmaxima = find_peaks(gradient_sum[row], distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])

                if argmaxima[0] < 48:  # maxima at left half
                    lane_boundary2_startpoint = np.array([[0, row]])
                else:
                    lane_boundary2_startpoint = np.array([[96, row]])

                lanes_found = True

            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1], row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0]) ** 2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]], 0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]], 0]])
                lanes_found = True

            row += 1

            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0, 0]])
                lane_boundary2_startpoint = np.array([[0, 0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def lane_detection(self, state_image_full):
        """
        This function should perform the road detection

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        """

        # to gray
        gray_state = self.cut_gray(state_image_full)  # gray state upside down

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        (
            lane_boundary1_points,
            lane_boundary2_points,
            lane_found,
        ) = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            #  in every iteration:
            # 1- find maximum/edge with the lowest distance to the last lane boundary point
            # 2- append maximum to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left
            #    or if the distance to the next one is too big (>=100)

            # lane_boundary 1
            # Iterate over each row after the first row found
            for row_index in range(lane_boundary1_points[0][1] + 1, len(maxima)):
                # Get the maxima for the current row
                row_maxima = maxima[row_index]

                # If there are no more maxima left or the distance to the next one is too far, break the loop
                if (
                    (row_maxima == 0).all()
                    or np.abs(row_maxima - lane_boundary1_points[-1][0]).min() >= 100
                    or (
                        lane_boundary1_points[-1][0] == 0
                        or lane_boundary1_points[-1][0] == 95
                    )
                ):
                    break

                # Find the closest index from the last lane boundary point
                closest_index = np.abs(
                    row_maxima - lane_boundary1_points[-1][0]
                ).argmin()
                closest_maxima = row_maxima[closest_index]

                if lane_boundary1_points[-1][0] < abs(
                    closest_maxima - lane_boundary1_points[-1][0]
                ):
                    closest_maxima = 0
                elif 95 - lane_boundary1_points[-1][0] < abs(
                    closest_maxima - lane_boundary1_points[-1][0]
                ):
                    closest_maxima = 95

                # Append this to lane_boundary1_points
                lane_boundary1_points = np.vstack(
                    (lane_boundary1_points, [[closest_maxima, row_index]])
                )

                if closest_maxima != 0 and closest_maxima != 95:
                    # Delete this maxima from maxima
                    maxima[row_index] = np.append(
                        np.delete(row_maxima, closest_index), 0
                    )

            # lane_boundary 2
            for row_index in range(lane_boundary2_points[0][1] + 1, len(maxima)):
                # Get the maxima for the current row
                row_maxima = maxima[row_index]

                # If there are no more maxima left or the distance to the next one is too far, break the loop
                if (
                    (row_maxima == 0).all()
                    or np.abs(row_maxima - lane_boundary2_points[-1][0]).min() >= 100
                    or (
                        lane_boundary2_points[-1][0] == 0
                        or lane_boundary2_points[-1][0] == 95
                    )
                ):
                    break

                # Find the closest index from the last lane boundary point
                closest_index = np.abs(
                    row_maxima - lane_boundary2_points[-1][0]
                ).argmin()
                closest_maxima = row_maxima[closest_index]

                if lane_boundary2_points[-1][0] < abs(
                    closest_maxima - lane_boundary2_points[-1][0]
                ):
                    closest_maxima = 0
                elif 95 - lane_boundary2_points[-1][0] < abs(
                    closest_maxima - lane_boundary2_points[-1][0]
                ):
                    closest_maxima = 95

                # Append this to lane_boundary2_points
                lane_boundary2_points = np.vstack(
                    (lane_boundary2_points, [[closest_maxima, row_index]])
                )

            # spline fitting using scipy.interpolate.splprep
            # and the arguments self.spline_smoothness
            #
            # if there are more lane_boundary points points than spline parameters
            # else use preceding spline
            if (
                lane_boundary1_points.shape[0] > 4
                and lane_boundary2_points.shape[0] > 4
            ):
                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                lane_boundary1_points = np.unique(
                    lane_boundary1_points, axis=0
                )  # remove duplicates
                lane_boundary1, _ = splprep(
                    lane_boundary1_points.T, s=self.spline_smoothness
                )  # compute spline

                # lane_boundary 2
                lane_boundary2_points = np.unique(
                    lane_boundary2_points, axis=0
                )  # remove duplicates
                lane_boundary2, _ = splprep(
                    lane_boundary2_points.T, s=self.spline_smoothness
                )  # compute spline
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        """
        Plot lanes and way points
        """
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))

        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(
            lane_boundary1_points_points[0],
            lane_boundary1_points_points[1] + 96 - self.cut_size,
            linewidth=5,
            color="orange",
        )
        plt.plot(
            lane_boundary2_points_points[0],
            lane_boundary2_points_points[1] + 96 - self.cut_size,
            linewidth=5,
            color="orange",
        )
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1] + 96 - self.cut_size, color="white")

        plt.axis("off")
        plt.xlim((-0.5, 95.5))
        plt.ylim((-0.5, 95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
