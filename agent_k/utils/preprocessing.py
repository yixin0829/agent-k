import re

hard_coded_mapping = {
    "NI 43-101 Technical Report for the Läntinen Koillismaa project in Finland dated May, 2021.pdf": "läntinen koillismaa",
    "Preliminary Economic Assessment (PEA) of the Tamarack North Project – Tamarack Minnesota": "tamarack north",
    "Updated Preliminary Economic Assessment on the Tamarack North Project": "tamarack north",
    "Preliminary Economic Assessment (PEA) #3 of the Tamarack North Project – Tamarack Minnesota": "tamarack north",
    "First Independent Technical Report on the Tamarack North Project": "tamarack north",
    "Rattlesnake Mtn. - Little Rattlesnake Mtn.": "little rattlesnake mountain",
    "NI 43-101 Technical Report - Ban Phuc Nickel Project": "ban phuc",
    "Nickel Mine": "maine",
    "O & C Land": "o&c land",
    "O&C Lands": "o&c land",
}


def preprocess_ms_name(ms_name: str) -> str:
    if ms_name in hard_coded_mapping:
        return hard_coded_mapping[ms_name]

    ms_name = ms_name.lower()
    # Extract mineral site name using regex
    # Example: 'NI 43-101 Technical Report for the Shakespeare project in Canada dated January, 2022' -> 'Shakespeare'
    # Example: 'NI 43-101 Technical Report for the Mel Project in North America dated July 2004' -> 'Mel'
    # Example: 'NI 43-101 Technical Report - Ban Phuc Nickel Project' -> 'Ban Phuc'
    if "43-101" in ms_name:
        # look behind for "for the " or "of the " and look ahead for " project"
        regex = r"(?:(for|of|on) the )(?P<ms_name>[a-zA-Z(-|\s)]+)(?= project)"
        match = re.search(regex, ms_name)
        ms_name = match.group("ms_name") if match else ""
    ms_name = re.sub(
        r"(mines|mine|prospects|prospect|deposits|deposit|nickel|copper|pge|manganese)",
        "",
        ms_name,
    )

    # Replace "mtn." with "mountain"
    ms_name = re.sub(r"(mtn|mt)\.", "mountain", ms_name)

    # First, replace multiple hyphens surrounded by whitespace with a single space
    ms_name = re.sub(r"\s+-+\s+", " ", ms_name)
    # Then, remove any remaining standalone hyphens at the start or end
    ms_name = re.sub(r"^-+\s+|\s+-+$", "", ms_name)

    # Remove any digits with whitespace around
    ms_name = re.sub(r"\s\d+\s", "", ms_name)

    ms_name = re.sub(r"\s+", " ", ms_name)
    ms_name = ms_name.strip()
    return ms_name


# Preprocess the mineral site names
def preprocess_ms_names(ms_list: list[str]) -> list[str]:
    ms_processed = []
    for i, ms_name in enumerate(ms_list):
        ms_processed.append(preprocess_ms_name(ms_name))
    # Deduplicate the list
    ms_processed = list(set(ms_processed))

    return ms_processed
