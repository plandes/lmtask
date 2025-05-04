from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.lmtask",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    package_data={'': ['*.conf', '*.json', '*.yml']},
    description='Large langauage model experimentation.',
    user='plandes',
    project='lmtask',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
