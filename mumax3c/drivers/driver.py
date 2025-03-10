import abc
import pathlib

import discretisedfield as df
import micromagneticmodel as mm
import ubermagtable as ut
import ubermagutil as uu

import mumax3c as mc


class Driver(mm.ExternalDriver):
    """Driver base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, "evolver"):
            self.autoselect_evolver = False
        else:
            self.autoselect_evolver = True

    @abc.abstractmethod
    def _checkargs(self, **kwargs):
        """Abstract method for checking arguments."""

    def drive_kwargs_setup(self, abspath=True, **kwargs):
        self._checkargs(**kwargs)
        kwargs.setdefault("abspath", abspath)

        # TODO OOMMF support additional arguments; are there equivalent options in mumax
        # fixed_subregions = None  # mumax?
        # compute = None  # mumax?
        # output_step = False  # mumax?

    def schedule_kwargs_setup(self, abspath=True, **kwargs):
        self._checkargs(**kwargs)
        kwargs.setdefault("abspath", abspath)

    def _write_input_files(self, system, **kwargs):
        self.write_mx3(system, **kwargs)

    def write_mx3(self, system, dirname=".", ovf_format="bin8", abspath=True, **kwargs):
        """Write the mx3 file and related files.

        Takes ``micromagneticmodel.System`` and write the mx3 file (and related files)
        to drive it in the phase space. The files are written directly to directory
        ``dirname`` (if not specified the current working directory). No additional
        subdirectiories are created.

        This method accepts any other arguments that could be required by the
        specific driver.

        Users are generally not encouraged to use this method directly. Instead
        ``Driver.drive``, ``Driver.schedule``, or ``mc.schedule`` should be used to
        write the files an run the simulation. This method is provided to give advanced
        users full flexibility.

        Parameters
        ----------
        system : micromagneticmodel.System

            System object to be driven.

        dirname : str, optional

            Name of a directory in which the input files are stored.
            If not specified the current workinng
            directory is used.

        ovf_format : str

            TODO UPDATE TO THE SUPPORTED MUMAX FILE TYPES
            Format of the magnetisation output files written by mumax3. Can be
            one of ``'bin8'`` (binary, double precision), ``'bin4'`` (binary,
            single precision) or ``'txt'`` (text-based, double precision).
            Defaults to ``'bin8'``.

        abspath : bool, optional

            If ``abspath=True`` absolute paths for additional input files (e.g. initial
            magnetisation) are written to the mx3 file. If ``abspath=False`` only
            filenames are written to the file. Relative files require mumax3 to be run
            from inside the directory containing the mx3 and other input files.

        """
        with uu.changedir(dirname):
            mx3 = mc.scripts.system_script(
                system, ovf_format=ovf_format, abspath=abspath
            )
            mx3 += mc.scripts.driver_script(
                self,
                system,
                compute=None,  # TODO does mumax3 support compute?
                ovf_format=ovf_format,
                **kwargs,
            )
            with open(self._mx3filename(system), "wt", encoding="utf-8") as mx3file:
                mx3file.write(mx3)

            # Generate and save json info file for a drive (not compute).
            if True:  # compute is None:  # TODO does mumxa3 support compute?
                self._write_info_json(system, **kwargs)

        # TODO if self/system is modified for mx3 creation reset it here
        delattr(system, "region_relator")

    def _call(self, system, runner, verbose=1, dry_run=False, **kwargs):
        if runner is None:
            runner = mc.runner.runner
        if dry_run:
            return runner.call(argstr=self._mx3filename(system), dry_run=True)
        else:
            runner.call(
                argstr=self._mx3filename(system),
                verbose=verbose,
                total=kwargs.get("n"),
                glob_name=f"{system.name}.out/m_full*.ovf",
            )

    def _read_data(self, system):
        # Update system's magnetisation. An example .ovf filename: m_full000000.ovf
        ovffiles = pathlib.Path(f"{system.name}.out").glob("m_full*.ovf")
        lastovffile = sorted(ovffiles)[-1]
        # pass Field.array instead of Field for better performance
        # Mumax3 norm changes so need to set back to old norm
        norm_field = system.m.norm
        system.m.value = df.Field.fromfile(str(lastovffile)).array
        system.m.norm = norm_field

        system.table = ut.Table.fromfile(
            str(pathlib.Path(f"{system.name}.out/table.txt")), x=self._x
        )

    @staticmethod
    def _mx3filename(system):
        return f"{system.name}.mx3"
