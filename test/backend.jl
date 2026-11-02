# The device the tests run on: this machine's GPU, Mantle's default device. Library calls take
# it as an argument (`BACKEND`).
import Mantle

const TEST_DEVICE = Mantle.Device()
const BACKEND = TEST_DEVICE
