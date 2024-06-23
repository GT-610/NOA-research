# Nutcracker Optimization Algorithm Python implemention
This is a Python implemention of Nutcracker Optimization Algorithm (NOA), a novel nature-inspired metaheuristic
algorithm for global optimization and engineering design problems.

## Run the code
1. Install the required packages:
    ```py
    pip install -r requirements.txt
    ```

    If you would like to calculate with GPU, you need to install `cupy`. See the [official documentation](https://docs.cupy.dev/en/stable/install.html) for install instructions.

    But please note that GPU calculation code is still **EXPERIMENTAL**.

2. Adjust the parameters
    
    Parameters are in the `main.py` (CPU calculation) or `main_cupy.py` (GPU calculation) with comments. See the comments in the codes for details.

3. Run the code
    * For CPU calculation:
    ```py
    python main.py
    ```

    * For GPU ~~accelerated~~ calculation (EXPERIMENTAL):
    ```py
    python main_cupy.py
    ```

    By default, the results are in `results` folder.

## Contributing
If there's a bug or any other problems, please start an issue or submit a PR.

## Credits
* [MATLAB implemention of NOA](https://github.com/redamohamed8/Nutcracker-Optimization-Algorithm) 

## License
MIT License

## Reference
[Mohamed Abdel-Basset, Reda Mohamed, Mohammed Jameel, Mohamed Abouhawwash,
Nutcracker optimizer: A novel nature-inspired metaheuristic algorithm for global optimization and engineering design problems,
Knowledge-Based Systems,
Volume 262,
2023,
110248,
ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2022.110248.
(https://www.sciencedirect.com/science/article/pii/S0950705122013442)
](https://doi.org/10.1016/j.knosys.2022.110248)