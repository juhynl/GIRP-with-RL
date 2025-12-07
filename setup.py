from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl_girp",
    version="0.1.0",
    author="Ah-hyun Lee, Yoonsang Han, Juhyeon Lee",
    author_email="ahlee230@sogang.ac.kr, han14931@sogang.ac.kr, ljuh0928@sogang.ac.kr",
    description="A reinforcement learning tools to play Bennett Foddy's GIRP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juhynl/GIRP-with-RL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">= 3.8",
    install_requires=["torch", "numpy", "selenium", "tensorboard"],
)
