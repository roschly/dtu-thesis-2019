
import config


class ConfigWriter:
    def __init__(self):
        pass

    @staticmethod
    def _get_config_vars_str() -> str:
        # Write all variables in config to file
        config_vars = [v for v in dir(config) if not v.startswith("__")]
        string_config_vars = [v + ": " + str(eval( "config." + str(v) )) for v in config_vars]
        return "\n".join(string_config_vars)

    @staticmethod
    def write_to_file():
        config_note = "Insert note here"
        config_string = ConfigWriter._get_config_vars_str()
        with open(config.experiment_folder_path + "config.txt", "w", newline="") as config_file:
            config_file.write("Note: " + config_note + "\n\n")
            config_file.write( config_string + "\n\n" )

            # encoder.summary(print_fn=lambda x: config_file.write(x + '\n'))
            # decoder.summary(print_fn=lambda x: config_file.write(x + '\n'))
            # full_model.summary(print_fn=lambda x: config_file.write(x + '\n'))




