import setuptools

if __name__ == "__main__":
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        author_email="clement.grisi@radboudumc.nl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/clemsgrs/prism-embedder",
        project_urls={"Bug Tracker": "https://github.com/clemsgrs/prism-embedder/issues"},
        packages=setuptools.find_packages(exclude=["tests"]),
        exclude_package_data={"": ["tests"]},
        entry_points={
            "console_scripts": [
                "prism_embedder = prism_embedder.main:run",
            ],
        },
    )