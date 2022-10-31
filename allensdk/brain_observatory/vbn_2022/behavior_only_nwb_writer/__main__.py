from allensdk.brain_observatory.vbn_2022.\
    behavior_only_nwb_writer.create_behavior_only_nwb import (
        VBN2022BehaviorOnlyWriter)


def main():
    writer = VBN2022BehaviorOnlyWriter()
    writer.run()


if __name__ == "__main__":
    main()
