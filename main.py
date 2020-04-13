from hand_gesture import hand_gesture as hg


def main():
    hand_ges = hg.HandGesture(source=0,size=(640,420))
    while True:
        hand_ges.detect_hand(lower_skin=[0, 20, 70], upper_skin=[20, 255, 255])
        hand_ges.recognize()
        hand_ges.display()
        flag = hand_ges.stop("q")
        if flag:
            break

    hand_ges.close_all()


if __name__ == "__main__":
    main()
