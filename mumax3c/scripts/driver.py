import discretisedfield as df
import micromagneticmodel as mm
import numpy as np

import mumax3c as mc
from .util import _identify_subregions
import warnings
import numbers


def driver_script(driver, system, compute=None, ovf_format="bin4", **kwargs):
    mx3 = "tableadd(E_total)\n"
    mx3 += "tableadd(dt)\n"
    mx3 += "tableadd(maxtorque)\n"
    if isinstance(driver, mc.MinDriver):
        for attr, value in driver:
            if attr != "evolver":
                mx3 += f"{attr} = {value}\n"

        mx3 += "minimize()\n\n"
        mx3 += "save(m_full)\n"
        mx3 += "tablesave()\n\n"

    if isinstance(driver, mc.RelaxDriver):
        if not system.dynamics.get(type=mm.Damping):
            raise ValueError("A damping term is needed.")
        alpha = system.dynamics.get(type=mm.Damping)[0].alpha
        mx3 += f"alpha = {alpha}\n"

        for attr, value in driver:
            if attr != "evolver":
                mx3 += f"{attr} = {value}\n"

        mx3 += "relax()\n\n"
        mx3 += "save(m_full)\n"
        mx3 += "tablesave()\n\n"

    if isinstance(driver, mc.TimeDriver):
        # Extract dynamics equation parameters.
        gamma0 = (precession[0].gamma0 if (precession := system.dynamics.get(
            type=mm.Precession)) else 0)
        if system.dynamics.get(type=mm.Damping):
            alpha = system.dynamics.damping.alpha
        else:
            alpha = 0

        mx3 += f"alpha = {alpha}\n"
        if not gamma0:
            mx3 += "doprecess = false\n"
        else:
            mx3 += f"gammaLL = {gamma0/mm.consts.mu0}\n"
            mx3 += "doprecess = true\n"

        if system.dynamics.get(type=mm.ZhangLi):
            mx3 += "// ZhangLi term\n"
            (zh_li_term, ) = system.dynamics.get(type=mm.ZhangLi)
            u = (zh_li_term.u
                 if isinstance(zh_li_term.u, df.Field) else df.Field(
                     mesh=system.m.mesh,
                     dim=3,
                     value=(1.0, 0.0, 0.0),
                     norm=zh_li_term.u,
                 ))

            j = -np.multiply(
                u * (mm.consts.e / (mm.consts.e * mm.consts.hbar /
                                    (2.0 * mm.consts.me))),
                system.m.norm,
            )
            j.write("j.ovf", representation=ovf_format)
            mx3 += f"Xi = {zh_li_term.beta}\n"
            mx3 += "Pol = 1\n"  # Current polarization is 1.
            mx3 += 'J.add(LoadFile("j.ovf"), 1)\n'  # 1 means constant in time.

        if system.dynamics.get(type=mm.Slonczewski):
            mx3 += "// STT term\n"
            mx3 += f"DisableZhangLiTorque = true\n"
            warnings.warn(
                f"STT supported with cross-direction respective to P only.")
            (slonczewski_term, ) = system.dynamics.get(type=mm.Slonczewski)

            def __region_defined_quantity_fallback(system, quant,
                                                   quant_name: str):
                reg_str = ""
                for reg in quant:
                    quant_val = quant[reg]
                    if quant_name == "J":
                        # exception for J, which is a vector quantity
                        quant_val = (0, 0, quant[reg])
                    if isinstance(quant_val, tuple) or isinstance(
                            quant_val, list):
                        quant_val = f"vector{quant_val}"
                    reg_indx = system.region_relator[reg][0]
                    reg_str += f"{quant_name}.setregion({reg_indx}, {quant_val})\n"
                return reg_str

            def __globally_defined_quantity_fallback(quant, quant_name: str):
                return f"{quant_name} = {quant}\n"

            def __quant_definition_dispatch(quant, quant_name: str, system):
                if isinstance(quant, df.Field):
                    raise ValueError(
                        f"Slonczewski term: {quant_name} with spatially varying parameters is not supported."
                    )
                elif isinstance(quant, dict):
                    return __region_defined_quantity_fallback(
                        system, quant, quant_name)
                else:
                    return __globally_defined_quantity_fallback(
                        quant, quant_name)

            def __quant_definition():
                mx3_ = ""
                mx3_ += __quant_definition_dispatch(slonczewski_term.mp,
                                                    "FixedLayer", system)
                mx3_ += __quant_definition_dispatch(slonczewski_term.Lambda,
                                                    "Lambda", system)
                mx3_ += __quant_definition_dispatch(slonczewski_term.P, "Pol",
                                                    system)
                mx3_ += __quant_definition_dispatch(slonczewski_term.eps_prime,
                                                    "EpsilonPrime", system)
                mx3_ += __quant_definition_dispatch(slonczewski_term.J, "J",
                                                    system)
                return mx3_

            mx3 += __quant_definition()

        mx3 += "relax()\n"
        t, n = kwargs["t"], kwargs["n"]
        dt = t/n
        mx3 += "setsolver(5)\n"
        mx3 += f"fixDt = {dt}\n\n"
        # mx3 += f"autosave(m, {dt})\n"
        # mx3 += f"tableautosave({dt})\n"
        # mx3 += f"run({t})\n"
        mx3 += f"for snap_counter:=0; snap_counter<{n}; snap_counter++{{\n"
        mx3 += f"    run({t/n})\n"
        mx3 +=  "    save(m_full)\n"
        mx3 +=  "    tablesave()\n"
        mx3 +=  "}"

    return mx3
