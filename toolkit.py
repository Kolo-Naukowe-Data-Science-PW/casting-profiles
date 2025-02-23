import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plotImage(image):
    """
    Plots black and white picture.
    :param image: two-dimensional array filled with values from 0 to 1
    :return: None
    """
    plt.imshow(X=image, cmap="binary")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plotImageWithGrid(image):
    """
    Plots black and white picture with grid
    :param image: two-dimensional array filled with values from 0 to 1
    :return: None
    """
    plt.imshow(X=image, cmap="binary")
    plt.tight_layout()
    plt.grid("on")
    plt.show()

def saveImage(image, path):
    """
    Saves figure as .png/.jpg
    :param image: 2-dimensional array filled with values from 0 to 1
    :param path: path to the destination with name of the output file
    :return:
    """
    print("do zrobienia")

def easy_case(data, empty_rows_threshold=10, empty_cols_threshold=10, scalling_coef_rows=0.04, scalling_coef_cols=0.04):
    """
    Checks if it is possible to separate the label with horizontal/vertical line,
    under the assumption is that the label is in the northwest part of the picture.
    :param data: 2-dimensional matrix
    :param empty_rows_threshold: rows with a sum of elements lower than that are find empty
    :param empty_cols_threshold: cols with a sum of elements lower than that are find empty
    :param scalling_coef_rows: proportion of half of the label to the whole picture
    :param scalling_coef_cols: proportion of half of the label to the whole picture
    :return: boolean value if algorithm worked
    """

    rowSums, colSums = data.sum(axis=1), data.sum(axis=0)

    emptyRows = (rowSums < empty_rows_threshold)
    firstNonEmptyRowIndex = np.argmin(emptyRows)
    if firstNonEmptyRowIndex < data.shape[0] // 2:
        labelSpaceHeight = np.argmax(emptyRows[firstNonEmptyRowIndex+int(data.shape[0]*scalling_coef_rows):])
        if labelSpaceHeight + firstNonEmptyRowIndex + int(data.shape[0]*scalling_coef_rows) < data.shape[0] // 3:
            data[:labelSpaceHeight + firstNonEmptyRowIndex + int(data.shape[0]*scalling_coef_rows), :] = 0
            return True

    emptyCols = colSums < empty_cols_threshold
    firstNonEmptyColIndex = np.argmin(emptyCols)
    if firstNonEmptyColIndex < data.shape[1] // 2:
        labelSpaceWidth = np.argmax(emptyCols[firstNonEmptyColIndex+int(data.shape[1]*scalling_coef_cols):])
        if labelSpaceWidth + firstNonEmptyColIndex + int(data.shape[1]*scalling_coef_cols) < data.shape[1] // 3:
            data[:, :labelSpaceWidth + firstNonEmptyColIndex + int(data.shape[1]*scalling_coef_cols)] = 0
            print(firstNonEmptyColIndex, labelSpaceWidth)
            print(colSums[labelSpaceWidth + firstNonEmptyColIndex-10: labelSpaceWidth + firstNonEmptyColIndex+10])
            print(emptyCols[labelSpaceWidth + firstNonEmptyColIndex-10: labelSpaceWidth + firstNonEmptyColIndex+10])
            return True

    return False

def read_and_convert_to_binary(filepath, threshold = 0.3, rows_sum_threshold=5, cols_sum_threshold=5):
    """
    Reads the image from file, get rid of noise and return 0, 1 matrix.
    :param filepath:
    :param threshold: Fields with value under this threshold are find empty.
    :param rows_sum_threshold: If a row's sum is lower than it the row is filled with zeros.
    :param cols_sum_threshold: If a col's sum is lower than it the col is filled with zeros.
    :return:
    """
    image = Image.open(filepath)

    data = np.asarray(image)
    data = 0.2989 * data[:,:,0] + 0.5870 * data[:,:,1] + 0.1140 * data[:,:,2]
    data = data / 255
    data = 1 - data

    rowSums = data.sum(axis=1)
    colSums = data.sum(axis=0)

    data[rowSums < rows_sum_threshold, :] = 0
    data[:, colSums < cols_sum_threshold] = 0


    return data > threshold

def handle_labels_01(data, row_diff_threshold=40, col_diff_threshold=50):
    """
    Searches row-wise and col-wise for the first occurrences of decreasing row/col-sum
    :param data:
    :param row_diff_threshold: First occurrence of difference between two consecutive rows sums lesser than -row_diff_threshold is considered as end of label (vertical).
    :param col_diff_threshold: First occurrence of difference between two consecutive columns sums lesser than -col_diff_threshold is considered as end of label (horizontal).
    :return:
    """

    if easy_case(data):
        return data

    rowSums, colSums = data.sum(axis=1), data.sum(axis=0)

    rowDiffs = np.diff(rowSums)
    colDiffs = np.diff(colSums)

    rowDiffs = (rowDiffs < -row_diff_threshold).astype(np.int8)
    colDiffs = (colDiffs < -col_diff_threshold).astype(np.int8)

    rowDiffs = np.diff(rowDiffs)
    colDiffs = np.diff(colDiffs)

    data[:int(np.argmin(rowDiffs)*1.05), :int(np.argmin(colDiffs)*1.05)] = 0

    return data

def handle_labels_02(data, coeff = 0.4):
    """
    Searches for first occurrence of difference between two consecutive (row/col)sums less than coeff*max_sum_through_dimension. Each step the threshold is divided by 2.
    """
    if easy_case(data):
        return data

    rowSums, colSums = data.sum(axis=1), data.sum(axis=0)

    rowDiffs = np.diff(rowSums)
    colDiffs = np.diff(colSums)
    maxRowDiff = np.max(np.abs(rowDiffs))
    maxColDiff = np.max(np.abs(colDiffs))

    _coeff = coeff
    row = col = 0
    for i in range(1000):


        _rowDiffs = (rowDiffs < -_coeff*maxRowDiff).astype(np.int8)
        _colDiffs = (colDiffs < -_coeff*maxColDiff).astype(np.int8)

        _rowDiffs = np.diff(rowDiffs)
        _colDiffs = np.diff(colDiffs)

        if np.any(_rowDiffs) and np.any(_colDiffs):
            break

        _coeff *= 0.5

    else:
        print("Invalid data.")
        return data

    data[:int(np.argmin(rowDiffs)*1.05), :int(np.argmin(colDiffs)*1.05)] = 0

    return data

def handle_labels_03(data, label_coeff=0.3):
    if easy_case(data):
        return data

    rowSums, colSums = data.sum(axis=1), data.sum(axis=0)
    rowDiffs = np.diff(rowSums)
    colDiffs = np.diff(colSums)
    rowDiffMax = np.max(np.abs(rowDiffs[:data.shape[0]//2]))
    colDiffMax = np.max(np.abs(colDiffs[:data.shape[0]//2]))


    for i in np.linspace(label_coeff, 0, 100):


        _rowDiffs = (rowDiffs < -i*rowDiffMax).astype(np.int8)
        _colDiffs = (colDiffs < -i*colDiffMax).astype(np.int8)

        _rowDiffs = np.diff(rowDiffs)
        _colDiffs = np.diff(colDiffs)

        if np.where(rowDiffs < 0)[0].shape[0] * np.where(colDiffs < 0)[0].shape[0] != 0:
            break
    else:
        print("Wystapił błąd")
        return data

    data[:int(np.where(rowDiffs)[0][-1]*1.05), :int(np.where(colDiffs)[0][-1]*1.05)] = 0
    return data

def reshape(data, finalHeight = 1754, finalWidth = 1275, reshape_margin=0.5, reshape_coeff=5e-6):
    rowSums, colSums = data.sum(axis=1), data.sum(axis=0)
    rowIndexes = colIndexes = np.array([])

    for i in range(1000):
        if rowIndexes.shape[0] * colIndexes.shape[0] != 0:
            break
        rows = rowSums >= reshape_coeff*np.max(rowSums)
        cols = colSums >= reshape_coeff*np.max(colSums)
        reshape_coeff *= 0.5

        colIndexes = np.where(cols)[0]
        rowIndexes = np.where(rows)[0]
    else:
        print(rowIndexes.shape, colIndexes.shape)
        print("Wystapił błąd")
        return np.zeros((finalHeight, finalWidth))

    rowFirst, rowLast = rowIndexes[1], rowIndexes[-1]
    colFirst, colLast = colIndexes[1], colIndexes[-1]
    rowWidth = rowLast - rowFirst
    colWidth = colLast - colFirst

    rowStart = max(int(rowFirst-reshape_margin*rowWidth), 0)
    colStart = max(int(colFirst-reshape_margin*colWidth), 0)
    rowEnd = min(int(rowLast+reshape_margin*rowWidth), data.shape[0])
    colEnd = min(int(colLast+reshape_margin*colWidth), data.shape[1])

    data = data[rowStart:rowEnd, colStart:colEnd]

    paddingTop = max(0, (finalHeight - data.shape[0]) // 2)
    paddingBot = max(0, finalHeight - paddingTop - data.shape[0])
    paddingLeft = max(0, (finalWidth - data.shape[1]) // 2)
    paddingRight = max(0, finalWidth - paddingLeft - data.shape[1])

    data = np.pad(
        1 - data, # to save in proper colours
        ((paddingTop, paddingBot), (paddingLeft, paddingRight)),
        mode="constant", constant_values=1
                 )

    return data


