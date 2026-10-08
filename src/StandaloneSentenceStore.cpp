#include "Util.hpp"
#include "SentenceStore.hpp"

// Standalone utilities have no parent IB database. The integrated application
// supplies its own bridge implementation; keep this fallback out of the library.
std::unique_ptr<SentenceStore> SentenceStoreFactory::CreateBridgeStore(void*)
{
    throw std::runtime_error("An IB bridge is unavailable in this standalone utility");
}
