from pathlib import Path
from typing import Optional, Sequence, Union

import py3Dmol
from IPython.display import Markdown, display
from opi.core import Calculator
from opi.input.blocks import Block, BlockDocker
from opi.input.blocks.base import InputFilePath
from opi.input.structures import Structure
from rdkit.Chem.rdchem import Atom, BondType, Conformer, GetPeriodicTable, RWMol
from rdkit.Geometry import Point3D


def show_structure(
    structure: Union[Structure, str, Path],
    title: str = "Structure",
    stick_radius: float = 0.1,
    sphere_scale: float = 0.2,
    label_level: int = 2,
    wireframe_indices: list[int] | None = None,
) -> None:
    """
    Visualize a Structure or XYZ file with optional atom/bond labels and wireframe styling using py3Dmol.

    Parameters:
        structure (Structure | str | Path): Structure object or path to .xyz file.
        title (str): Title to display.
        stick_radius (float): Stick style bond radius.
        sphere_scale (float): Atom sphere scaling.
        label_level (int):
            0 = no labels,
            1 = show atom symbol + index,
            2 = as above, plus bond lengths.
        wireframe_indices (list[int]): Atom indices whose spheres will be downsized to stick radius.
    """
    if isinstance(structure, (str, Path)):
        path = Path(structure).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        structure = Structure.from_xyz(path)

    xyz_block = structure.to_xyz_block().splitlines()
    num_atoms = int(xyz_block[0])
    atom_lines = xyz_block[2 : 2 + num_atoms]

    atoms = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    mol = RWMol()
    for symbol in atoms:
        mol.AddAtom(Atom(symbol))
    conf = Conformer(len(atoms))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf)

    # Infer bonds
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
            ri = GetPeriodicTable().GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
            rj = GetPeriodicTable().GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
            if dist < 1.3 * (ri + rj):
                mol.AddBond(i, j, BondType.SINGLE)
    mol = mol.GetMol()

    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(structure.to_xyz_block(), "xyz")
    viewer.setBackgroundColor("white")

    # Global style
    viewer.setStyle(
        {"stick": {"radius": stick_radius}, "sphere": {"scale": sphere_scale}}
    )

    # Override wireframe atoms
    if wireframe_indices:
        viewer.setStyle(
            {"serial": [i for i in wireframe_indices]},
            {
                "stick": {"radius": stick_radius},
                "sphere": {"scale": 0.0},  # override to match stick radius
            },
        )

    # Atom labels
    if label_level >= 1:
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            label = f"{atom.GetSymbol()}{idx}"
            viewer.addLabel(
                label,
                {
                    "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "fontSize": 18,
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "showBackground": False,
                    "id": f"{title.lower()}_atom_{idx}",
                },
            )

    # Bond length labels
    if label_level == 2:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            p1 = conf.GetAtomPosition(i)
            p2 = conf.GetAtomPosition(j)
            mid = {
                "x": (p1.x + p2.x) / 2,
                "y": (p1.y + p2.y) / 2,
                "z": (p1.z + p2.z) / 2,
            }
            dist = p1.Distance(p2)
            viewer.addLabel(
                f"{dist:.2f}",
                {
                    "position": mid,
                    "fontSize": 18,
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "showBackground": False,
                },
            )

    viewer.zoomTo()
    display(Markdown(f"### {title}"))
    viewer.show()


def setup_calc(
    working_dir: Optional[Union[Path, str]],
    structure: Union[Structure, str],
    basename: str = "single_molecule_calc",
    sk_list: Optional[Sequence[str]] = None,
    bk_list: Optional[Union[Block, Sequence[Block]]] = None,
    ncores: int = 12,
    memory: int = 2200,
    visualize: bool = True,
) -> Calculator:
    """
    Set up a single-molecule ORCA calculation using a Structure, path to XYZ file, or SMILES string.

    Parameters:
        working_dir (Path or str or None): Directory for input/output files.
        structure (Structure or path or SMILES): The molecule.
        basename (str): Base name for calculation.
        sk_list (list[str]): Simple keywords for ORCA input.
        bk_list (Block or list[Block] or None): Additional ORCA input blocks.
        ncores (int): Number of CPU cores.
        memory (int): Memory in MB.
        visualize (bool): Whether to visualize the structure.

    Returns:
        Calculator: Configured Calculator instance.
    """
    # Normalize and create working directory
    if isinstance(working_dir, str):
        working_dir = Path(working_dir)
    if working_dir is None:
        working_dir = Path.cwd() / basename
    working_dir = working_dir.resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    # Convert input to Structure
    if isinstance(structure, Structure):
        mol = structure
    elif isinstance(structure, str):
        path = Path(structure)
        if path.suffix == ".xyz" and path.exists():
            mol = Structure.from_xyz(path)
        else:
            # Treat as SMILES
            mol = Structure.from_smiles(structure)

    # Save structure to XYZ
    xyz_file = working_dir / f"{basename}.xyz"
    xyz_file.write_text(mol.to_xyz_block())

    # Optional visualization
    if visualize:
        show_structure(mol, title=basename)

    # Set up Calculator
    calc = Calculator(basename=basename, working_dir=working_dir)
    calc.structure = mol
    calc.structure.charge = mol.charge
    calc.structure.multiplicity = mol.multiplicity

    if sk_list:
        calc.input.add_simple_keywords(*sk_list)

    # Add blocks
    if isinstance(bk_list, Block):
        calc.input.add_blocks(bk_list)
    elif isinstance(bk_list, Sequence):
        for bk in bk_list:
            if not isinstance(bk, Block):
                raise TypeError(f"Invalid block in bk_list: {bk}")
            calc.input.add_blocks(bk)

    # Resources
    calc.input.ncores = ncores
    calc.input.memory = memory

    calc.write_input()
    return calc


def setup_docking_calc(
    working_dir: Union[str, Path],
    host_smiles: str,
    guest_smiles: str,
    basename: str = "docking_calc",
    sk_list: Optional[Sequence[str]] = ["xtb", "alpb(water)"],
    bk_list: Optional[Sequence[Block]] = None,
    ncores: int = 12,
    memory: int = 2200,
    visualize: bool = True,
) -> Calculator:
    """
    Set up an ORCA docking calculation using host and guest SMILES strings.

    Parameters:
        working_dir (str or Path): Base directory where 'docking' subdirectory will be created.
        host_smiles (str): SMILES string of the host molecule.
        guest_smiles (str): SMILES string of the guest molecule.
        basename (str): Base name for input/output files.
        sk_list (list[str]): Simple ORCA keywords.
        bk_list (list[Block] or None): Additional input blocks.
        ncores (int): Number of CPU cores.
        memory (int): Memory in MB.
        visualize (bool): Whether to visualize host and guest structures.

    Returns:
        Calculator: Configured Calculator instance.
    """
    # Normalize and prepare paths
    if isinstance(working_dir, str):
        working_dir = Path(working_dir)
    docking_dir = working_dir.resolve() / "docking"
    docking_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save guest structure
    guest_structure = Structure.from_smiles(guest_smiles)
    guest_xyz_path = docking_dir / f"{guest_smiles.lower()}.xyz"
    guest_xyz_path.write_text(guest_structure.to_xyz_block())

    if visualize:
        show_structure(guest_structure, title="Guest")

    # Prepare Docker block
    docker_block = BlockDocker(guest=InputFilePath.from_string(guest_xyz_path.name))

    # Combine with optional blocks to
    all_blocks: list[Block] = [docker_block]
    if bk_list:
        all_blocks.extend(bk_list)

    # Use setup_calc to build host-side calculator with guest block
    calc = setup_calc(
        working_dir=docking_dir,
        structure=host_smiles,
        basename=basename,
        sk_list=sk_list,
        bk_list=all_blocks,
        ncores=ncores,
        memory=memory,
        visualize=visualize,
    )
    return calc
