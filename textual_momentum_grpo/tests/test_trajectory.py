from tmgrpo.trajectory import TrajectoryState, generate_momentum, generate_textual_gradient, update_digest


class FakeClient:
    """Stands in for LLMClient: records calls, returns scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system_prompt, user_prompt, temperature=1.0):
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def test_generate_textual_gradient_calls_client_with_summary():
    client = FakeClient(["diagnosis text"])
    result = generate_textual_gradient(client, "step summary here")
    assert result == "diagnosis text"
    assert client.calls[0][1] == "step summary here"


def test_update_digest_handles_empty_current_digest():
    client = FakeClient(["updated digest"])
    result = update_digest(client, "", "new gradient")
    assert result == "updated digest"
    assert "empty" in client.calls[0][1].lower()


def test_generate_momentum_passes_digest():
    client = FakeClient(["try X next"])
    result = generate_momentum(client, "digest content")
    assert result == "try X next"
    assert client.calls[0][1] == "digest content"


def test_trajectory_state_step_updates_digest_and_momentum():
    client = FakeClient(["gradient-1", "digest-1", "momentum-1"])
    state = TrajectoryState()
    assert state.momentum == ""  # M_0 = empty, per README section 3 step 1

    m1 = state.step(client, "step 1 summary")

    assert m1 == "momentum-1"
    assert state.momentum == "momentum-1"
    assert state.digest == "digest-1"
    assert state.history == ["gradient-1"]


def test_trajectory_state_step_is_incremental_not_concatenating():
    # Second step's digest-update call must see the FIRST digest as its "current digest" context,
    # not the raw list of all past gradients concatenated (README section 3, step 5).
    client = FakeClient(
        ["gradient-1", "digest-1", "momentum-1", "gradient-2", "digest-2", "momentum-2"]
    )
    state = TrajectoryState()
    state.step(client, "step 1 summary")
    state.step(client, "step 2 summary")

    # Call order: gradient,digest,momentum,gradient,[digest] -- index 4 is step 2's digest update.
    digest_update_call = client.calls[4]
    assert "digest-1" in digest_update_call[1]
    assert state.digest == "digest-2"
    assert state.history == ["gradient-1", "gradient-2"]
