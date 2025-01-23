import contextlib
import os
import re
from abc import abstractmethod

from dotenv import load_dotenv

from san2patch.utils.cmd import BaseCommander


class BaseReproducer(BaseCommander):
    def __init__(self, vuln_id: str, project_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        load_dotenv(override=True)

        self.vuln_id = vuln_id
        self.project_name = project_name

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def reproduce(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


class VulnlocReproducer(BaseReproducer):
    name = "vulnloc"

    def __init__(self, vuln_data, *args, **kwargs):
        super().__init__(vuln_data["bug_id"], vuln_data["subject"], *args, **kwargs)

        self.main_dir = os.path.join(os.getenv("DATASET_EXTRACTED_DIR"), self.name)

        self.binary_path: str = vuln_data["binary_path"]
        self.crash_input: str = vuln_data["crash_input"]
        if len(vuln_data["exploit_file_list"]) == 0:
            self.exploit_file = None
        else:
            self.exploit_file: str = vuln_data["exploit_file_list"][0].split("/")[-1]

        self.san_raw_dir = os.path.join(os.getenv("DATASET_EXTRACTED_DIR"), self.name, "sanitizer_raw")
        self.stdout_file = os.path.join(self.san_raw_dir, f"{self.vuln_id}.out")
        self.stderr_file = os.path.join(self.san_raw_dir, f"{self.vuln_id}.err")

        self.data_dir = f"/home/yuntong/vulnloc-benchmark/{self.project_name}/{self.vuln_id}"
        self.experiment_dir = f"/experiment/vulnloc-benchmark/{self.project_name}/{self.vuln_id}"

    def setup(self):
        self.container_id = self.run_cmd('docker ps -qf "name=vulnloc"', cwd=self.main_dir, pipe=True).stdout.strip()

        if self.container_id == None or self.container_id == "":
            raise ValueError("Vulnloc container not found. Please run the container first.")

            # Pull docker and run
            self.run_cmd(f"docker pull yuntongzhang/vulnfix:latest-manual", cwd=self.main_dir)
            self.run_cmd(
                f"docker run -itd --memory=30g --name vulnfix yuntongzhang/vulnfix:latest-manual /bin/bash",
                cwd=self.main_dir,
            )

            # Get docker id
            self.container_id = self.run_cmd(
                'docker ps -qf "name=vulnfix"', cwd=self.main_dir, pipe=True
            ).stdout.strip()

        # Download vulnloc-benchmark
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd /home/yuntong && cd vulnloc-benchmark || git clone https://github.com/nus-apr/vulnloc-benchmark"',
            cwd=self.main_dir,
        )

        self.logger.debug(f"Container ID: {self.container_id}")

    def build(self):
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./setup.sh"',
            cwd=self.main_dir,
        )
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./config.sh"',
            cwd=self.main_dir,
        )
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./build.sh"',
            cwd=self.main_dir,
        )

    def reproduce(self):
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./test.sh {self.exploit_file}"',
            cwd=self.main_dir,
        )

        docker_vuln_out_file = os.path.join(self.experiment_dir, "src", self.binary_path + ".out")
        host_vuln_out_file = os.path.join(self.san_raw_dir, f"{self.vuln_id}.err")

        # Copy sanitizer output to the host
        self.run_cmd(
            f"docker cp {self.container_id}:{docker_vuln_out_file} {host_vuln_out_file}",
            cwd=self.main_dir,
        )

        with open(host_vuln_out_file, "r") as f_stderr:
            stderr = f_stderr.read()

            sanitizer_re_1 = r"ERROR: .+Sanitizer:"
            sanitizer_re_2 = r"SUMMARY: .+Sanitizer:"
            sanitizer_re_3 = r"runtime error:"

            # Check if the sanitizer is found
            if (
                re.search(sanitizer_re_1, stderr)
                or re.search(sanitizer_re_2, stderr)
                or re.search(sanitizer_re_3, stderr)
            ):
                self.logger.success("Sanitizer detected the crash.")
                self.logger.success("Reproduce completed ans success.")
                return True
            else:
                self.logger.error("Crash not found.")
                self.logger.error("Reproduce failed.")
                return False

    def run(self):
        self.setup()
        self.build()
        self.reproduce()


class San2PatchReproducer(BaseReproducer):
    name = "san2patch"

    def __init__(self, vuln_data, *args, **kwargs):
        super().__init__(vuln_data["bug_id"], vuln_data["subject"], *args, **kwargs)

        self.main_dir = os.path.join(os.getenv("DATASET_EXTRACTED_DIR"), self.name)

        self.binary_path: str = vuln_data["binary_path"]
        self.crash_input: str = vuln_data["crash_input"]
        if len(vuln_data["exploit_file_list"]) == 0:
            self.exploit_file = None
        else:
            self.exploit_file: str = vuln_data["exploit_file_list"][0].split("/")[-1]

        self.san_raw_dir = os.path.join(os.getenv("DATASET_EXTRACTED_DIR"), self.name, "sanitizer_raw")
        self.stdout_file = os.path.join(self.san_raw_dir, f"{self.vuln_id}.out")
        self.stderr_file = os.path.join(self.san_raw_dir, f"{self.vuln_id}.err")

        self.data_dir = f"/home/san2patch/san2patch-benchmark/{self.project_name}/{self.vuln_id}"
        self.experiment_dir = f"/experiment/san2patch-benchmark/{self.project_name}/{self.vuln_id}"

    def setup(self):
        self.container_id = self.run_cmd('docker ps -qf "name=san2patch"', cwd=self.main_dir, pipe=True).stdout.strip()

        if self.container_id == None or self.container_id == "":
            raise ValueError("San2Patch container not found. Please run the container first.")

        # Download vulnloc-benchmark
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd /home/san2patch && cd san2patch-benchmark || git clone https://github.com/san2patch/san2patch-benchmark"',
            cwd=self.main_dir,
        )

        self.logger.debug(f"Container ID: {self.container_id}")

    def build(self):
        # self.run_cmd(
        #     f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./setup.sh"',
        #     cwd=self.main_dir,
        # )
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./config.sh"',
            cwd=self.main_dir,
        )
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./build.sh"',
            cwd=self.main_dir,
        )

    def reproduce(self):
        self.run_cmd(
            f'docker exec {self.container_id} bash -c "cd {self.data_dir} && ./test.sh {self.exploit_file}"',
            cwd=self.main_dir,
        )

        docker_vuln_out_file = os.path.join(self.experiment_dir, "src", self.binary_path + ".out")
        host_vuln_out_file = os.path.join(self.san_raw_dir, f"{self.vuln_id}.err")

        # Copy sanitizer output to the host
        self.run_cmd(
            f"docker cp {self.container_id}:{docker_vuln_out_file} {host_vuln_out_file}",
            cwd=self.main_dir,
        )

        with open(host_vuln_out_file, "r") as f_stderr:
            stderr = f_stderr.read()

            sanitizer_re_1 = r"ERROR: .+Sanitizer:"
            sanitizer_re_2 = r"SUMMARY: .+Sanitizer:"
            sanitizer_re_3 = r"runtime error:"

            # Check if the sanitizer is found
            if (
                re.search(sanitizer_re_1, stderr)
                or re.search(sanitizer_re_2, stderr)
                or re.search(sanitizer_re_3, stderr)
            ):
                self.logger.success("Sanitizer detected the crash.")
                self.logger.success("Reproduce completed ans success.")
                return True
            else:
                self.logger.error("Crash not found.")
                self.logger.error("Reproduce failed.")
                return False

    def run(self):
        self.setup()
        self.build()
        self.reproduce()
