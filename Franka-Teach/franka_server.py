from frankateach.franka_server import FrankaServer
import hydra

@hydra.main(version_base="1.2", config_path="configs", config_name="franka_server")
def main(cfg):
    fs = FrankaServer(cfg.deoxys_config_path)
    fs.init_server()


if __name__ == "__main__":
    main()