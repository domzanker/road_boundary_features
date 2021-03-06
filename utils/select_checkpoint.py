from pathlib import Path
import ruamel.yaml as yaml


def get_best_checkpoint(path) -> Path:
    # return best checkpoint file in dir
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        file_list = []
        for file in path.iterdir():
            if file.suffix == ".ckpt":
                file_list.append(file)

        if len(file_list) == 0:
            return None
        elif len(file_list) == 1:
            return file_list[0]
        else:
            # list all available inputs
            print("More than one available checkpoint. Please select")
            str = ["[%s] %s" % (i, p) for i, p in enumerate(file_list)]
            str.append(
                'Select file [%s,..,%s] or "q" to abort: ' % (0, len(file_list) - 1)
            )
            selected = input("\n".join(str))
            if selected == "q":
                return None
            else:
                try:
                    return file_list[int(selected)]
                except ValueError:
                    return None


if __name__ == "__main__":

    with open("params.yaml", "rb") as f:
        configs = yaml.round_trip_load(f)

    with open("data/checkpoint.ckpt", "w+") as f:
        pass

    if configs["train"]["load_weights"] or configs["train"]["resume_training"]:
        best_checkpoint = get_best_checkpoint(configs["train"]["checkpoint_path"])
        # move checkpoint to stage
        if best_checkpoint is not None:
            configs["train"]["checkpoint_path"] = str(best_checkpoint)
            best_checkpoint.rename("data/checkpoint.ckpt")

    with open("params.yaml", "w") as f:
        yaml.round_trip_dump(
            configs,
            f,
            indent=4,
            block_seq_indent=4,
        )
