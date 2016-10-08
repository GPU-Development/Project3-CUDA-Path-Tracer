#pragma once

namespace StreamCompaction {
namespace Efficient {
	void scan_dev(int n, int *dev_data);
	int compact_dev(int n, int *dev_out, const int *dev_in);
}
}
