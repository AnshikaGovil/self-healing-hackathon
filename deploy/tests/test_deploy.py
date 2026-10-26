"""Offline tests for the deployment scripts.

The cloud calls themselves are not run here - no Modal, no Lightning, no
network - only the decisions the scripts make: what a GPU minute costs, when
a cap is reached, how the accounts file is read.

    python deploy/tests/test_deploy.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import bis_cloud                                               # noqa: E402
import budget                                                  # noqa: E402
def test_the_accounts_file_reads_like_a_dotenv():
    """One file holds every account the deployment uses, so switching to a
    colleague's is an edit to it - comments, blank lines, quotes and an
    `export` in front all read the way a person expects."""
    import tempfile
    path = os.path.join(tempfile.mkdtemp(prefix="bis-acc-"), "accounts.env")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Modal\n\nMODAL_TOKEN_ID=ak-123\n"
                     "MODAL_TOKEN_SECRET = 'as-456'\n"
                     'export HF_TOKEN="hf_789"\n'
                     "BIS_BUDGET_USD=3.00   # this month\n"
                     "LIGHTNING_API_KEY=\n"
                     "not a setting\n")
    acc = bis_cloud.load_accounts(path)
    assert acc["MODAL_TOKEN_ID"] == "ak-123", acc
    assert acc["MODAL_TOKEN_SECRET"] == "as-456", acc
    assert acc["HF_TOKEN"] == "hf_789", acc
    assert acc["BIS_BUDGET_USD"] == "3.00", acc
    assert acc["LIGHTNING_API_KEY"] == "", acc
    assert "not a setting" not in " ".join(acc), acc
    assert bis_cloud.load_accounts(path + ".missing") == {}


def test_a_key_is_written_back_in_place():
    text = "# keys\nBIS_ADMIN_KEY=\nOTHER=1\n"
    out = bis_cloud.set_value(text, "BIS_ADMIN_KEY", "abc")
    assert out == "# keys\nBIS_ADMIN_KEY=abc\nOTHER=1\n", out
    out = bis_cloud.set_value(out, "NEW_KEY", "x")
    assert out.endswith("NEW_KEY=x\n") and "BIS_ADMIN_KEY=abc" in out, out


def test_the_deployed_addresses_are_read_from_modal_s_output():
    """Modal names an endpoint after the workspace, so the addresses are
    read back from each deploy - another account, another address."""
    out = ("Created Function Judge.*.\n    Judge.serve -> \n"
           "        https://someone--bis-judge-judge-serve.us-east.modal.run"
           " \n        (unauthenticated)\n"
           "web => https://someone--bis-healing.us-east.modal.run\n")
    assert bis_cloud.JUDGE_URL.search(out).group(0) == (
        "https://someone--bis-judge-judge-serve.us-east.modal.run")
    assert bis_cloud.FRONT_URL.search(out).group(0) == (
        "https://someone--bis-healing.us-east.modal.run")
    assert bis_cloud.FRONT_URL.search("https://x--bis-judge.modal.run") \
        is None


def test_a_notebook_joins_the_front_door_with_a_pasted_cell():
    """Kaggle (and a Lightning Studio) run the same model server, which
    registers itself with the front door. The cell names the front door and
    where it stands in the failover - and carries no key: those come from
    the platform's own secret store."""
    front = "https://someone--bis-healing.us-east.modal.run"
    cell = bis_cloud.kaggle_cell(front)
    for want in (front, "--register", "--name kaggle", "--priority 3",
                 "--max-hours 11", "UserSecretsClient", "HF_TOKEN",
                 "BIS_ADMIN_KEY", "kaggle_ai_server.py", "cu124"):
        assert want in cell, (want, cell)
    shell = bis_cloud.lightning_cell(front, hours=4)
    for want in (front, "--register", "--name lightning", "--priority 2",
                 "--max-hours 4", "--platform Lightning", "read -s"):
        assert want in shell, (want, shell)
    compile(cell, "kaggle-cell", "exec")


def test_modal_is_driven_with_the_account_the_file_names():
    """A token pair wins over a profile, a profile over this machine's
    default login; nothing named, nothing changed."""
    both = bis_cloud.modal_env({"MODAL_TOKEN_ID": "ak", "MODAL_TOKEN_SECRET":
                                "as", "MODAL_PROFILE": "someone"})
    assert both == {"MODAL_TOKEN_ID": "ak", "MODAL_TOKEN_SECRET": "as"}, both
    assert bis_cloud.modal_env({"MODAL_PROFILE": "someone"}) == {
        "MODAL_PROFILE": "someone"}
    assert bis_cloud.modal_env({"MODAL_TOKEN_ID": "ak"}) == {}, (
        "half a token pair is no account")
    assert bis_cloud.modal_env({}) == {}


def test_a_gpu_second_is_charged_at_its_rate_and_then_some():
    """The ledger charges what the GPU costs plus the CPU, memory and boot
    the container bills beside it, so it reaches the cap before the
    account's credit does, never after."""
    hour = budget.charge(0.0, 3600, "T4")
    assert hour > 0.59, hour
    assert hour < 1.0, hour
    assert budget.charge(1.0, 3600, "T4") == 1.0 + hour
    assert budget.charge(0.0, 3600, "L4") > hour
    assert budget.charge(2.5, 0, "T4") == 2.5
    assert budget.charge(2.5, -30, "T4") == 2.5, "time never runs backwards"
    assert budget.charge(0.0, 60, "A100") > budget.charge(0.0, 60, "L4"), (
        "a card with no rate is charged as the dearest one known, not free")


def test_the_front_door_s_cpu_is_charged_too():
    """The front door runs on Modal's CPUs, billed from the same credit as
    the GPU, so its hours go into the same ledger: one core and 2 GiB, about
    $0.06 an hour, more with the margin."""
    hour = budget.charge(0.0, 3600, "CPU")
    assert 0.06 < hour < 0.12, hour
    assert hour < budget.charge(0.0, 3600, "T4") / 5


def test_a_container_is_charged_its_boot_as_well():
    """Modal bills a container from its boot; the ledger starts when the
    container's code does. Measured on the first deployment, three
    containers: Modal $0.062, the ledger without a boot allowance $0.053 -
    the boots (about 15-25 s each, and the whole short life of a capped
    one) were the difference. 30 s a container covers it."""
    assert budget.BOOT_S >= 30
    three = budget.charge(0.0, 3 * budget.BOOT_S + 290, "T4")
    assert three >= 0.062, three


def test_the_cap_is_reached_at_the_cap_and_no_cap_means_none():
    assert not budget.over(2.99, 3.0)
    assert budget.over(3.0, 3.0)
    assert budget.over(3.5, 3.0)
    assert budget.over(0.0, 0.0), "a cap of 0 refuses everything"
    assert budget.over(0.0, -1.0)


def test_each_month_keeps_its_own_ledger():
    """Modal's credit is monthly, so the ledger is too."""
    assert budget.month_key(0) == "spent-1970-01"
    assert budget.month_key(1798761600) == "spent-2027-01"


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as exc:
            failures += 1
            print("  FAIL  %s  %s" % (name, exc))
        except Exception as exc:
            failures += 1
            print("  ERROR %s  %s: %s" % (name, type(exc).__name__, exc))
    print("\n%s" % ("all tests passed" if not failures
                    else "%d test(s) failed" % failures))
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
