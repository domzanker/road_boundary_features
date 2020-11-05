import ruamel.yaml as yaml


def yesno(prompt, invalid_prompt="Please type [y(es), n(o)]"):
    ask = True
    while ask:
        answer = input(prompt)
        if answer in ["y", "yes"]:
            return True
        elif answer in ["n", "no"]:
            return False
        else:
            prompt = invalid_prompt


def prompt_gpu(gpu_confs):
    print("Experiment configured tp use GPU: %s" % gpu_confs, "\r")
    prompt = "Change settings? [y/n] "
    invalid_prompt = "Invalid input. Please type [y(es)/ n(o)] "
    return yesno(prompt, invalid_prompt)


def add_gpu(gpu_confs):
    inp = input("Enter GPUs: [#, none]")
    if len(inp) == 0 or inp == "none":
        return 0
    if "," in inp:
        split = inp.split(",")
    else:
        split = inp.split(" ")
    return list(map(int, split))


def prompt_tags():
    if yesno("Do you want to add tags to the experiment? [y/n]"):
        tags = input("Tag: ")
        if "," in tags:
            return tags.split(",")
        else:
            return tags.split(" ")


def name(name):
    if yesno("Current name for experiment: %s. Want to change? [y/n]" % name):
        return input("Type name: ")
    else:
        return name


if __name__ == "__main__":
    with open("experiment.yaml", "r") as f:
        configs = yaml.round_trip_load(f)

    configs["tags"] = []
    if prompt_gpu(configs["gpu"]):
        gpu = add_gpu(configs["gpu"])
        configs["gpu"] = gpu
    configs["name"] = name(configs["name"])
    configs["tags"] = prompt_tags()

    with open("experiment.yaml", "w") as f:
        yaml.round_trip_dump(configs, f, indent=4, block_seq_indent=4)
