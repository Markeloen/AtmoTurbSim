from setuptools import setup, find_packages

setup(
    name='my_package',  # Name of your package
    version='0.1.0',    # Version of your package
    author='Your Name',  # Your name
    author_email='your.email@example.com',  # Your email
    description='A short description of the package',  # A short description
    long_description=open('README.md').read(),  # Long description read from the readme
    long_description_content_type='text/markdown',  # Type of the long description
    url='http://github.com/yourusername/my_package',  # Link to your project's repository
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        'numpy',  # Dependencies for your package
        'pandas'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Development status
        'Intended Audience :: Developers',  # Target audience
        'License :: OSI Approved :: MIT License',  # License
        'Programming Language :: Python :: 3',  # Supported Python versions
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',  # Minimum version requirement of the Python interpreter
)
