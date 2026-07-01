class EdgeDevice:
    def __init__(self, name, mac, domain, analysis_type, data_type):
        self.name: str = name
        self.mac: str = mac
        self.domain: str = domain
        self.analysis_type: str = analysis_type
        self.data_type: str = data_type

    def to_config_file(self):
        return self._generate_config_file(vars(self))

    def _generate_config_file(self, variables: dict[str, str]):
        lines = []
        for key, value in variables.items():
            lines.append(f"{key}={value}")
        return "\n".join(lines) + "\n"        