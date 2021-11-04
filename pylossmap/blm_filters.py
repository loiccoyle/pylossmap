class Filters:
    """Generic filtering class, make sure to define a self.filter and a
    self._blm_list_filter.
    """

    @staticmethod
    def _sanitize_inp(inp, prepare=None, check=None):
        if check is not None:
            if not set(inp) <= check:
                raise ValueError(f"Input must be subset of {check}.")

        if callable(prepare):
            inp = map(prepare, inp)
        else:
            inp = map(str, inp)
        return "|".join(inp)

    def DS(self, mask=False):
        """Selects the BLMs in the dispersion suppressor region.

        Args:
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the dispersion suppressors BLMs.
        """
        return self.filter(rf"BLMQ[IE]\.(0[7-9]|10|11)[RL][37]", mask=mask)

    def IR(self, *IRs, mask=False):
        """Filters the BLMs based on the IR(s).

        Args:
            *IRs (int): IR(s) of interests.
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array with the filtered IR(s).
        """
        IR = self._sanitize_inp(IRs, check={1, 2, 3, 4, 5, 6, 7, 8})
        return self.filter(rf"\.\d\d[LR]({IR})", mask=mask)

    def TCP(self, HVS=False, mask=False):
        """Selects only the TCP BLMs.

        Args:
            HVS (bool, optional): If True will filter for the BLM of
                Horizontal, Vertical and Skew collimators.
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the TCP BLMs.
        """
        if HVS:
            pattern = r"BLMTI.*TCP\."
        else:
            pattern = r"TCP\."
        return self.filter(pattern, mask=mask)

    def TCS(self, mask=False):
        """Selects only the TCS BLMs.

        Args:
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the TCS BLMs.
        """
        return self.filter(r"TCS[GP][M]?\.", mask=mask)

    def TCL(self, mask=False):
        """Selects only the TCL BLMs.

        Args:
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the TCS BLMs.
        """
        return self.filter(r"TCL[A]?\.", mask=mask)

    def TCTP(self, mask=False):
        """Selects only the TCTP BLMs.

        Args:
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the TCTP BLMs.
        """
        return self.filter(r"TCTP[HV]\.", mask=mask)

    def TCLI(self, mask=False):
        """Selects only the TCLI BLMs.

        Args:
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the TCLI BLMs.
        """
        return self.filter(r"TCLI[AB]\.", mask=mask)

    def side(self, RL, mask=False):
        """Filters the BLMs based on their side.

        Args:
            RL (str): Either "R" or "L" or "RL".
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the filtered BLMs.
        """
        return self.filter(rf"\.\d\d[{RL}][1-8]", mask=mask)

    def cell(self, *cells, mask=False):
        """Filters the BLMs based on their cell number(s).

        Args:
            *cells (int): cells of interest.
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the filtered cells.
        """

        def pad(x):
            return f"{x:02}" if x < 10 else str(x)

        cells = self._sanitize_inp(cells, prepare=pad)
        return self.filter(rf"\.({cells})[RL][1-8]", mask=mask)

    def beam(self, *beams, mask=False):
        """Filters the BLMs based on the beam(s).

        Args:
            *beam (int): Beams of interest, subset of {0,1,2}.
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the filtered beam(s).
        """
        beam = self._sanitize_inp(beams, check={0, 1, 2}, prepare=lambda x: str(int(x)))
        return self.filter(rf"B({beam})", mask=mask)

    def type(self, *types, mask=False):
        """Gets the BLM for the requested blm types.

        Args:
            types (str): string of the type(s) of interest, subset of
                {cold, warm, coll, xrp}.
            mask (bool, optional): if True will return a boolean mask array,
                otherwise will return a filtered LossMap/BLMData instance.

        Returns:
            LossMap/BLMData or boolean array: LossMap/BLMData instance or
                boolean mask array containing the desired BLM types.
        """
        allowed = {"cold", "warm", "coll", "xrp"}
        if not set(types) <= allowed:
            raise ValueError(f'"{types}" must be subset of {allowed}.')

        blm_list = self.meta[self.meta["type"].isin(types)].index.tolist()
        return self._blm_list_filter(blm_list)
